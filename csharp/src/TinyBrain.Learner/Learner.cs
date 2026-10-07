using System.Text.Json;

namespace TinyBrain.Learner;

/// <summary>What the learner claims about one region on one tick (exported for scoring; never read back).</summary>
public sealed record Claim(string Link, Epistemic Epistemic, int Group, List<int> Possible);

public sealed record IdentityEvent(string Kind, int Unit, int? Other, string InfId);

/// <summary>Per-tick output: which pixels each unit claims, and how the learner currently believes they relate.</summary>
public sealed class LearnerExport
{
    public required int Tick { get; init; }
    public required ushort[] Labels { get; init; }              // 0 = nothing claimed, else unit id
    public required Dictionary<int, Claim> Claims { get; init; }
    public required List<(int unit, string record)> RecordsMade { get; init; }
    public required List<IdentityEvent> IdentityEvents { get; init; }
}

/// <summary>
/// THE LEARNER. It receives only its calibration (once) and Observations (per tick). It never sees
/// the world, ground truth or any label: the TinyBrain.Learner assembly does not even reference the
/// TinyBrain.World assembly.
///
/// Required architecture implemented here (not experimental):
///   - observation store separate from inference store; provenance on every inference;
///   - persistent units created on first distinguishable encounter;
///   - experience records kept separately and addressable;
///   - provisional identities (possibly_same_as) and UNKNOWN as a first-class state;
///   - revisions as new inferences (history never erased); units never deleted;
///   - local learning, checked by a mutation audit.
/// Everything else is a replaceable Mechanism chosen in LearnerConfig.
/// </summary>
public sealed class Learner
{
    public Calibration Calib { get; }
    public LearnerConfig Config { get; }
    public ObservationStore Observations { get; } = new();
    public InferenceStore Inferences { get; } = new();
    public Dictionary<int, Unit> Units { get; } = new();
    public Dictionary<string, int> DecisionCounts { get; } = new() { ["SAME"] = 0, ["NEW"] = 0, ["UNKNOWN"] = 0 };
    public Dictionary<string, int> TouchLog { get; } = new();
    public int LocalityViolations { get; private set; }

    private readonly Dictionary<int, int> _parent = new();     // believed-identity union-find (from same_identity inferences)
    private Dictionary<int, int> _active = new();              // unit id -> candidate label on the previous tick
    private HashSet<int> _involved = new();
    private int _nextUnit = 1;
    private readonly RecognitionContext _rctx;

    public Learner(Calibration calib, LearnerConfig config)
    {
        Calib = calib;
        Config = config;
        _rctx = new RecognitionContext { Kinds = config.Evidence.ToDictionary(e => e.Name, e => e), Usefulness = config.Usefulness };
    }

    public int Find(int u)
    {
        while (_parent.TryGetValue(u, out var p) && p != u) u = p;
        return u;
    }

    private void Touch(Unit u, string reason)
    {
        if (!_involved.Contains(u.Id)) LocalityViolations++;
        TouchLog[reason] = TouchLog.GetValueOrDefault(reason) + 1;
    }

    private Unit NewUnit(int tick, string decision, string infId)
    {
        var u = new Unit(_nextUnit++, tick, decision, infId);
        Units[u.Id] = u;
        _parent[u.Id] = u.Id;
        _involved.Add(u.Id);
        Touch(u, "creation");
        return u;
    }

    /// <summary>Existing identity groups a query may be compared with (excluding units observed at the same moment).</summary>
    private List<(int, List<Unit>)> Groups(HashSet<int> exclude, int? olderThan = null)
    {
        var bad = exclude.Select(Find).ToHashSet();
        return Units.Values
            .Where(u => !exclude.Contains(u.Id) && u.Records.Count > 0 && (olderThan is null || u.CreatedTick < olderThan))
            .GroupBy(u => Find(u.Id))
            .Where(g => !bad.Contains(g.Key))
            .Select(g => (g.Key, g.ToList()))
            .ToList();
    }

    private ExperienceRecord MakeRecord(Unit u, int tick, string obsId, string link, Dictionary<string, object?> ev,
                                        string regionInf, string linkInf, int npix)
    {
        var inf = Inferences.Add(tick, "experience_record", "core:record", new[] { obsId, regionInf, linkInf }, Epistemic.Believed,
                                 new() { ["unit"] = u.Id, ["link"] = link });
        var r = new ExperienceRecord($"exp:{u.Id}:{u.Records.Count}", u.Id, tick, obsId, link, u.Segment, ev, inf.Id, npix);
        u.Records.Add(r);
        return r;
    }

    public LearnerExport Step(Observation obs)
    {
        int W = Calib.Width, H = Calib.Height;
        if (obs.Left.Length != W * H * 3 || obs.Right.Length != W * H * 3)
            throw new ArgumentException("observation does not match the calibration");
        int t = obs.Tick;
        _involved = new HashSet<int>();
        var M = Config;
        string oid = Observations.Add(obs);

        // ---- perception (all inferences, with provenance)
        M.SelfMotion.Step(Observations.SelfMotion(oid));
        Inferences.Add(t, "pose_estimate", M.SelfMotion.Tag, new[] { oid }, Epistemic.Believed,
                       new() { ["x"] = M.SelfMotion.X, ["z"] = M.SelfMotion.Z, ["psi"] = M.SelfMotion.Psi });
        var grayL = ImageOps.ToGray(Observations.Left(oid), W, H);
        var grayR = ImageOps.ToGray(Observations.Right(oid), W, H);
        var (disp, valid) = M.Stereo.Compute(grayL, grayR, W, H);
        var dispInf = Inferences.Add(t, "disparity_map", M.Stereo.Tag, new[] { oid }, Epistemic.Believed,
                                     new() { ["stored"] = false, ["recomputable_from"] = oid, ["n_valid"] = valid.Count(v => v) });
        var (labels, n, cr, cg, br) = M.Candidates.Extract(Observations.Left(oid), disp, valid, W, H);
        var cam = Geometry.CameraPoints(disp, valid, Calib);
        var ctx = new RegionContext
        {
            WorldPoints = Geometry.ToSelfFrame(cam, Calib, M.SelfMotion), CamPoints = cam,
            ChromaR = cr, ChromaG = cg, Bright = br, Gray = grayL, Calib = Calib,
        };
        var regionPix = new List<int>[n + 1];
        for (int c = 0; c <= n; c++) regionPix[c] = new List<int>();
        for (int i = 0; i < labels.Length; i++) if (labels[i] > 0) regionPix[labels[i]].Add(i);
        var pix = regionPix.Select(l => l.ToArray()).ToArray();
        var regionInf = new string[n + 1];
        for (int c = 1; c <= n; c++)
            regionInf[c] = Inferences.Add(t, "candidate_region", M.Candidates.Tag, new[] { dispInf.Id, oid }, Epistemic.Believed,
                                          new() { ["npix"] = pix[c].Length }).Id;
        var evCache = new Dictionary<int, Dictionary<string, object?>>();
        Dictionary<string, object?> EvidenceOf(int c) => evCache.TryGetValue(c, out var e) ? e
            : evCache[c] = M.Evidence.ToDictionary(k => k.Name, k => k.Extract(ctx, pix[c]));

        // ---- continuity
        var prev = _active.Keys.Select(u => (u, Units[u].LastPix)).ToList();
        var links = M.Continuity.Link(prev, labels, n, Observations.SelfMotion(oid).DYaw, Calib);
        var newActive = new Dictionary<int, int>();
        var claims = new Dictionary<int, string>();
        var made = new List<ExperienceRecord>();
        bool due = M.Sampling.Due(t);
        foreach (var (c, (uid, iou)) in links)
        {
            var u = Units[uid];
            _involved.Add(uid);
            Touch(u, "continuity");
            var link = Inferences.Add(t, "continuity_link", M.Continuity.Tag,
                                      new[] { regionInf[c], u.Records.Count > 0 ? u.Records[^1].InfId : u.CreationInf },
                                      Epistemic.Believed, new() { ["unit"] = uid, ["iou"] = Math.Round(iou, 3) });
            newActive[uid] = c;
            u.LastSeen = t; u.LastPix = pix[c];
            claims[uid] = "continuity";
            if (due) made.Add(MakeRecord(u, t, oid, "continuity", EvidenceOf(c), regionInf[c], link.Id, pix[c].Length));
        }

        // ---- candidates with no continuity: recognition (largest first)
        foreach (int c in Enumerable.Range(1, n).Where(c => !links.ContainsKey(c)).OrderByDescending(c => pix[c].Length))
        {
            var ev = EvidenceOf(c);
            var res = M.Recognition.Decide(new() { ev }, Groups(newActive.Keys.ToHashSet()), _rctx);
            DecisionCounts[res.Decision.ToString().ToUpper()]++;
            var dec = Inferences.Add(t, "recognition_decision", M.Recognition.Tag, new[] { regionInf[c] },
                                     res.Decision == Decision.Unknown ? Epistemic.Unknown : Epistemic.Believed,
                                     new() { ["decision"] = res.Decision.ToString(), ["group"] = res.Group, ["possible"] = res.Possible.Take(20).ToList(), ["detail"] = res.Detail });
            if (res.Decision == Decision.Same)
            {
                var u = Units[res.Unit!.Value];
                if (!newActive.ContainsKey(u.Id))
                {
                    _involved.Add(u.Id);
                    Touch(u, "recognition_link");
                    u.Segment++;                       // recognition never extends continuity authority
                    u.LastSeen = t; u.LastPix = pix[c];
                    newActive[u.Id] = c;
                    claims[u.Id] = "recognition";
                    made.Add(MakeRecord(u, t, oid, "recognition", ev, regionInf[c], dec.Id, pix[c].Length));
                    continue;
                }
                res = res with { Decision = Decision.Unknown, Possible = new() { Find(u.Id) } };  // cannot be two regions at once
            }
            var nu = NewUnit(t, res.Decision == Decision.New ? "NEW" : "PROVISIONAL", dec.Id);
            if (nu.Decision == "PROVISIONAL")
            {
                nu.Possible = res.Possible.ToList();
                Inferences.Add(t, "possibly_same_as", M.Recognition.Tag, new[] { dec.Id }, Epistemic.Unknown,
                               new() { ["unit"] = nu.Id, ["possible"] = nu.Possible.Take(20).ToList() });
            }
            nu.LastPix = pix[c];
            newActive[nu.Id] = c;
            claims[nu.Id] = "creation";
            made.Add(MakeRecord(nu, t, oid, "creation", ev, regionInf[c], dec.Id, pix[c].Length));
        }

        // ---- co-visibility: regions observed at the same moment are believed distinct
        foreach (var uid in newActive.Keys)
        {
            var u = Units[uid];
            int before = u.CoVisible.Count;
            u.CoVisible.UnionWith(newActive.Keys.Where(x => x != uid));
            if (u.CoVisible.Count != before) Touch(u, "co_visible");
        }

        // ---- usefulness samples: continuity (W) and simultaneous distinction (B) only — no feedback path
        var U = M.Usefulness;
        var changed = new Dictionary<int, Dictionary<string, (string set, double d, string rec, string partner)>>();
        void Add(Unit u, string set, string k, double d, string rec, string partner)
        {
            u.AddSample(set, k, new Sample(d, t, rec, partner));
            if (!changed.TryGetValue(u.Id, out var m)) changed[u.Id] = m = new();
            m[k] = (set, d, rec, partner);
        }
        foreach (var r in made.Where(r => r.Link == "continuity"))
        {
            var u = Units[r.Unit];
            var ds = U.Within(u, r, _rctx.Kinds);
            foreach (var (k, (d, partner)) in ds) Add(u, "W", k, d, r.Id, partner);
            if (ds.Count > 0) Touch(u, "usefulness");
        }
        var byUnit = made.ToDictionary(r => r.Unit, r => r);
        var ids = byUnit.Keys.OrderBy(x => x).ToList();
        for (int i = 0; i < ids.Count; i++)
            for (int j = i + 1; j < ids.Count; j++)
            {
                var (ua, ub) = (Units[ids[i]], Units[ids[j]]);
                foreach (var (uu, other) in new[] { (ua, byUnit[ids[j]]), (ub, byUnit[ids[i]]) })
                    foreach (var (k, (d, partner)) in U.Between(uu, other, _rctx.Kinds, t))
                        Add(uu, "B", k, d, other.Id, partner);
                Touch(ua, "usefulness"); Touch(ub, "usefulness");
            }
        foreach (var (uid, kinds) in changed)
        {
            var u = Units[uid];
            foreach (var (k, cause) in kinds)
            {
                double nw = U.Usefulness(u, k), old = u.Useful.GetValueOrDefault(k);
                if (Math.Abs(nw - old) <= 1e-9) continue;
                u.Useful[k] = nw;
                u.History.Add(new UsefulnessChange(t, k, old, nw, u.W.GetValueOrDefault(k)?.Count ?? 0, u.B.GetValueOrDefault(k)?.Count ?? 0,
                                                   cause.set, cause.d, cause.rec, cause.partner));
            }
        }

        // ---- revision of young identities (as new inferences; nothing erased)
        var events = new List<IdentityEvent>();
        if (M.Revision.Enabled)
            foreach (var r in made)
            {
                var u = Units[r.Unit];
                if (r.Link != "continuity" || u.LinkedToOlder || u.Records.Count > M.Revision.MaxRecords) continue;
                var groups = Groups(u.CoVisible.Append(u.Id).ToHashSet(), u.CreatedTick);
                if (groups.Count == 0) continue;
                var res = M.Recognition.Decide(u.Records.Select(x => x.Evidence).ToList(), groups, _rctx);
                var inputs = new[] { u.CreationInf }.Concat(u.Records.Select(x => x.InfId)).ToList();
                if (res.Decision == Decision.Same)
                {
                    var other = Units[res.Unit!.Value];
                    _involved.Add(u.Id); _involved.Add(other.Id);
                    var inf = Inferences.Add(t, "same_identity", M.Revision.Tag, inputs.Append(other.Records[^1].InfId), Epistemic.Believed,
                                             new() { ["units"] = new[] { u.Id, other.Id }, ["detail"] = res.Detail });
                    _parent[Find(u.Id)] = Find(other.Id);
                    u.LinkedToOlder = true; u.Open = false;
                    Touch(u, "identity_revision"); Touch(other, "identity_revision");
                    events.Add(new IdentityEvent("same", u.Id, other.Id, inf.Id));
                }
                else if (res.Decision == Decision.Unknown && !res.Possible.SequenceEqual(u.Possible))
                {
                    _involved.Add(u.Id);
                    var inf = Inferences.Add(t, "possibly_same_as", M.Revision.Tag, inputs, Epistemic.Unknown,
                                             new() { ["unit"] = u.Id, ["possible"] = res.Possible.Take(20).ToList(), ["supersedes_previous"] = true });
                    u.Possible = res.Possible.ToList(); u.Open = true;
                    Touch(u, "identity_revision");
                    events.Add(new IdentityEvent("possibly", u.Id, null, inf.Id));
                }
                else if (res.Decision == Decision.New && u.Open)
                {
                    _involved.Add(u.Id);
                    var inf = Inferences.Add(t, "resolved_distinct", M.Revision.Tag, inputs, Epistemic.Believed, new() { ["unit"] = u.Id });
                    u.Possible = new(); u.Open = false;
                    Touch(u, "identity_revision");
                    events.Add(new IdentityEvent("resolved_distinct", u.Id, null, inf.Id));
                }
            }

        _active = newActive;
        var outLabels = new ushort[W * H];
        var outClaims = new Dictionary<int, Claim>();
        foreach (var (uid, c) in newActive)
        {
            foreach (int p in pix[c]) outLabels[p] = (ushort)uid;
            var u = Units[uid];
            outClaims[uid] = new Claim(claims[uid], u.Open ? Epistemic.Unknown : Epistemic.Believed, Find(uid),
                                       u.Open ? u.Possible.Take(5).ToList() : new());
        }
        return new LearnerExport
        {
            Tick = t, Labels = outLabels, Claims = outClaims, IdentityEvents = events,
            RecordsMade = made.Select(r => (r.Unit, r.Id)).ToList(),
        };
    }

    /// <summary>Every W sample must come from a continuity record (no recognition feedback).</summary>
    public (int wSamples, int bSamples, int wNotFromContinuity) FeedbackAudit()
    {
        var link = Units.Values.SelectMany(u => u.Records).ToDictionary(r => r.Id, r => r.Link);
        int nw = 0, nb = 0, bad = 0;
        foreach (var u in Units.Values)
        {
            foreach (var l in u.W.Values) foreach (var s in l) { nw++; if (link.GetValueOrDefault(s.RecordId) != "continuity") bad++; }
            nb += u.B.Values.Sum(l => l.Count);
        }
        return (nw, nb, bad);
    }

    /// <summary>The learner's entire belief state as text (used by the harness to scan for leaked ground truth).</summary>
    public string StateDump() => JsonSerializer.Serialize(new
    {
        inferences = Inferences.Records.Select(r => new { r.Id, r.Tick, r.Kind, r.Mechanism, r.Inputs, Ep = r.Epistemic.ToString(), Payload = r.Payload.ToDictionary(kv => kv.Key, kv => kv.Value?.ToString()) }),
        units = Units.Values.Select(u => new { u.Id, u.CreatedTick, u.Decision, Records = u.Records.Select(x => x.Id), u.Possible, History = u.History.Count }),
    });
}
