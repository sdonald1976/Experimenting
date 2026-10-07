using TinyBrain.Learner;

namespace TinyBrain.World;

public sealed record Trial(int Cycle, int Phase, string Type, string Thing, bool? ExperiencedViewpoint, bool WasUnobserved, bool EventUnobserved, string Outcome);

public sealed record UnitSummary(int Unit, string Thing, int Records, int CreatedTick, Dictionary<string, double> LearnedUsefulness);

public sealed class VariantResult
{
    public required string Variant { get; init; }
    public List<Trial> Trials { get; } = new();
    public Dictionary<string, object> Audit { get; } = new();
    public Dictionary<string, int> DecisionCounts { get; set; } = new();
    public int Units { get; set; }
    public int Records { get; set; }
    public List<UnitSummary> MappedUnits { get; } = new();
    public List<string> Mechanisms { get; set; } = new();
}

public sealed class SeedResult
{
    public int Seed { get; init; }
    public int Ticks { get; set; }
    public List<string> CycleKinds { get; set; } = new();
    public List<string> SkippedEvents { get; } = new();
    public Dictionary<int, string> ThingNames { get; set; } = new();
    public Dictionary<string, VariantResult> Variants { get; } = new();
    public double WallSeconds { get; set; }
}

/// <summary>
/// Runs one seed of the Experiment 2 world against several learner variants (in this process; the learner
/// assembly cannot see this one). Per tick: apply hidden world events, render the stereo pair, send ONLY the
/// Observation to each learner, score each learner's export against ground truth.
/// </summary>
public static class ExperimentRun
{
    public const double SmNoiseRel = 0.03, SmNoiseAbsM = 0.003, SmNoiseAbsRad = 0.002;

    public static SeedResult Run(int seed, IReadOnlyList<string> variants, Action<string>? log = null,
                                 Action<int, byte[], byte[], int[], LearnerExport>? frame = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var world = new World(seed);
        var ticks = Schedule.Build(world);
        var rend = new Renderer(256, 192, 60, 0.12, -28 * Math.PI / 180, 1.0, 3.0);
        var calib = rend.Calibration;
        var noise = new Rng(30_000 + seed);
        var evRng = new Rng(40_000 + seed);
        var learners = variants.ToDictionary(v => v, v => new Learner.Learner(calib, LearnerConfig.Variant(v)));
        var scorers = variants.ToDictionary(v => v, _ => new VariantScorer());
        var recordGt = variants.ToDictionary(v => v, _ => new Dictionary<string, (int gt, double purity)>());
        var experiencedAz = new Dictionary<int, List<double>>();
        var sentHashes = new Dictionary<string, string>();
        var result = new SeedResult { Seed = seed, Ticks = ticks.Count };
        foreach (var v in variants) result.Variants[v] = new VariantResult { Variant = v, Mechanisms = learners[v].Config.Mechanisms.Select(m => m.Describe()).ToList() };

        // per-cycle bookkeeping
        Dictionary<string, Dictionary<int, int>>? snap = null;
        Dictionary<int, List<double>>? azPrior = null;
        (string kind, List<int> moved, List<int> swapped, int? added, bool unobserved) cur = ("", new(), new(), null, true);
        var invisible = new HashSet<int>();
        var slides = new List<(Thing t, double[] vel, int left)>();
        int[]? lastGt = null; TickPlan? prev = null; int obsIndex = 0, cycle = -1, phase = 1;

        for (int i = 0; i < ticks.Count; i++)
        {
            var tk = ticks[i];
            if (tk.EventKind is not null)
            {
                snap = variants.ToDictionary(v => v, v => scorers[v].AssociationSnapshot());
                azPrior = experiencedAz.ToDictionary(kv => kv.Key, kv => kv.Value.ToList());
                cur = ApplyEvent(world, tk.EventKind, tk.ReturnAlpha!.Value, evRng, experiencedAz, tk.Cycle!.Value, result);
                cycle = tk.Cycle!.Value; phase = tk.Phase;
                invisible.Clear();
                result.CycleKinds.Add(tk.EventKind);
            }
            if (tk.SlideStart && lastGt is not null) slides = StartSlides(world, evRng, lastGt);
            for (int s = 0; s < slides.Count; s++)
                if (slides[s].left > 0 && world.Things.Contains(slides[s].t))
                {
                    var (t, vel, left) = slides[s];
                    t.Pos = new[] { t.Pos[0] + vel[0], t.Pos[1] + vel[1] };
                    slides[s] = (t, vel, left - 1);
                }

            var (L, R, gt) = rend.Stereo(world.Surfaces, tk.Pos, tk.Psi, noise);
            lastGt = gt;
            if (tk.EventKind is not null)
            {
                var touched = cur.moved.Concat(cur.swapped).Concat(cur.added is int a ? new[] { a } : Array.Empty<int>());
                cur.unobserved = touched.All(g => !gt.Contains(g));
            }
            var visible = world.Things.ToDictionary(t => t.GtId, t => gt.Count(x => x == t.GtId));
            if (tk.Part == "away") foreach (var (g, n) in visible) if (n == 0) invisible.Add(g);
            foreach (var t in world.Things)
                if (visible[t.GtId] >= ScoringParams.VisibleMinPx)
                {
                    if (!experiencedAz.TryGetValue(t.GtId, out var l)) experiencedAz[t.GtId] = l = new();
                    l.Add(World.ObjectAzimuth(t, tk.Pos));
                }

            // noisy self-motion: the only non-visual input
            SelfMotion sm = new(0, 0, 0);
            if (prev is not null)
            {
                double fwdx = Math.Sin(prev.Psi), fwdz = Math.Cos(prev.Psi), rx = Math.Cos(prev.Psi), rz = -Math.Sin(prev.Psi);
                double dx = tk.Pos[0] - prev.Pos[0], dz = tk.Pos[1] - prev.Pos[1];
                double df = dx * fwdx + dz * fwdz, dr = dx * rx + dz * rz, dy = tk.Psi - prev.Psi;
                double N(double x, double ab) => x + noise.Normal(0, SmNoiseRel * Math.Abs(x) + ab);
                sm = new SelfMotion(N(df, SmNoiseAbsM), N(dr, SmNoiseAbsM), N(dy, SmNoiseAbsRad));
            }
            prev = tk;
            var obs = new Observation(i, L, R, sm);
            sentHashes[$"obs:{i}"] = ObservationStore.HashOf(obs);

            var exports = new System.Collections.Concurrent.ConcurrentDictionary<string, LearnerExport>();
            Parallel.ForEach(variants, v => exports[v] = learners[v].Step(obs));
            foreach (var v in variants)
            {
                var ex = exports[v];
                scorers[v].Update(ex, gt);
                foreach (var (uid, rid) in ex.RecordsMade)
                {
                    var vals = Enumerable.Range(0, gt.Length).Where(p => ex.Labels[p] == uid).Select(p => gt[p]).ToList();
                    if (vals.Count == 0) continue;
                    var best = vals.GroupBy(x => x).MaxBy(g => g.Count())!;
                    recordGt[v][rid] = (best.Key, best.Count() / (double)vals.Count);
                }
            }
            frame?.Invoke(i, L, R, gt, exports[variants[0]]);

            obsIndex = tk.Part == "observe" ? obsIndex + 1 : 0;
            if (tk.Part == "observe" && snap is not null && obsIndex - 1 == 9)      // evaluate after a 10-tick look
                foreach (var t in world.Things.Where(t => visible[t.GtId] >= ScoringParams.VisibleMinPx))
                {
                    int g = t.GtId;
                    string type = g == cur.added ? (cur.kind == "replace" ? "replace_new" : "novel")
                                : cur.swapped.Contains(g) ? "swapped" : cur.moved.Contains(g) ? "moved" : "unmoved";
                    bool? seenVp = null;
                    if (type is "swapped" or "moved" or "unmoved")
                    {
                        double az = World.ObjectAzimuth(t, tk.Pos);
                        seenVp = azPrior!.TryGetValue(g, out var hist) &&
                                 hist.Any(h => Math.Abs(((az - h + 540) % 360) - 180) <= ScoringParams.ViewpointToleranceDeg);
                    }
                    foreach (var v in variants)
                        result.Variants[v].Trials.Add(new Trial(cycle, phase, type, world.Names[g], seenVp, invisible.Contains(g), cur.unobserved,
                                                                scorers[v].Outcome(exports[v], gt, g, snap[v])));
                }
            if (i % 50 == 0) log?.Invoke($"seed {seed}: tick {i}/{ticks.Count}  phase {tk.Phase}  {tk.Part}");
        }

        result.ThingNames = new Dictionary<int, string>(world.Names);
        var leakTerms = new[] { SceneParams.Canary }.Concat(world.Names.Where(kv => kv.Key >= 10).Select(kv => kv.Value)).ToList();
        foreach (var v in variants)
        {
            var le = learners[v];
            var vr = result.Variants[v];
            var (nw, nb, bad) = le.FeedbackAudit();
            string dump = le.StateDump();
            vr.Audit["observations_unaltered"] = sentHashes.All(kv => le.Observations.Hashes.GetValueOrDefault(kv.Key) == kv.Value);
            vr.Audit["inferences_missing_provenance"] = le.Inferences.MissingProvenance;
            vr.Audit["locality_violations"] = le.LocalityViolations;
            vr.Audit["leaks_in_learner_state"] = leakTerms.Where(dump.Contains).ToList();
            vr.Audit["w_samples"] = nw; vr.Audit["b_samples"] = nb; vr.Audit["w_samples_not_from_continuity"] = bad;
            vr.DecisionCounts = new(le.DecisionCounts);
            vr.Units = le.Units.Count;
            vr.Records = le.Units.Values.Sum(u => u.Records.Count);
            // map long-lived units to the thing most of their records show (harness-only, for reading the output)
            foreach (var u in le.Units.Values.Where(u => u.Records.Count >= 10))
            {
                var labels = u.Records.Select(r => recordGt[v].TryGetValue(r.Id, out var x) && x.purity >= 0.5 ? x.gt : -99).ToList();
                var top = labels.GroupBy(x => x).MaxBy(g => g.Count())!;
                if (top.Key >= 10 && top.Count() >= 0.8 * labels.Count)
                    vr.MappedUnits.Add(new UnitSummary(u.Id, world.Names[top.Key], u.Records.Count, u.CreatedTick, new(u.Useful)));
            }
        }
        result.WallSeconds = sw.Elapsed.TotalSeconds;
        return result;
    }

    private static void FaceExperienced(Thing t, double retAlpha, Rng rng, Dictionary<int, List<double>> az)
    {
        var cam = World.ArcPos(retAlpha);
        double a = (az.TryGetValue(t.GtId, out var l) && l.Count > 0 ? rng.Choice(l) : 0) * Math.PI / 180;
        t.Yaw = Math.Atan2(cam[0] - t.Pos[0], cam[1] - t.Pos[1]) - a;
    }

    private static (string, List<int>, List<int>, int?, bool) ApplyEvent(World w, string kind, double retAlpha, Rng rng,
                                                                         Dictionary<int, List<double>> az, int cycle, SeedResult res)
    {
        var moved = new List<int>(); var swapped = new List<int>(); int? added = null;
        if (kind == "swap")
        {
            int i = rng.Integer(w.Things.Count), j;
            do { j = rng.Integer(w.Things.Count); } while (j == i);
            var (a, b) = (w.Things[i], w.Things[j]);
            (a.Pos, b.Pos) = (b.Pos, a.Pos);
            FaceExperienced(a, retAlpha, rng, az); FaceExperienced(b, retAlpha, rng, az);
            swapped.AddRange(new[] { a.GtId, b.GtId });
        }
        else if (kind == "move")
        {
            var order = w.Things.ToList(); rng.Shuffle(order);
            foreach (var t in order)
            {
                try { t.Pos = w.FreePosition(t.FootprintRadius, t, t.Pos); }
                catch (InvalidOperationException) { continue; }
                FaceExperienced(t, retAlpha, rng, az); moved.Add(t.GtId); break;
            }
        }
        else if (kind is "novel" or "replace")
        {
            Thing? victim = null; double[]? pos = null;
            if (kind == "replace") { victim = w.Things[rng.Integer(w.Things.Count)]; w.Things.Remove(victim); pos = victim.Pos; }
            for (int tries = 0; tries < 400 && added is null; tries++)
            {
                string shape = rng.Choice(new[] { "box", "cylinder", "sphere" }), col = rng.Choice(SceneParams.Palette.Keys.ToList());
                if (victim is not null && shape == victim.Shape && col == victim.Colour) continue;
                var size = World.RandomSize(rng, shape);
                double fp = World.Footprint(shape, size);
                double[] p;
                if (pos is null) { try { p = w.FreePosition(fp); } catch (InvalidOperationException) { continue; } }
                else if (w.IsFree(pos, fp)) p = pos; else continue;
                added = w.AddThing(shape, col, size, p, rng.Uniform(0, 2 * Math.PI), $"NOVEL{cycle}").GtId;
            }
            if (added is null)      // crowded floor: undo and record the skip (as in Experiment 2's harness)
            {
                if (victim is not null) w.Things.Add(victim);
                res.SkippedEvents.Add(kind);
            }
        }
        return (kind, moved, swapped, added, true);
    }

    private static List<(Thing, double[], int)> StartSlides(World w, Rng rng, int[] lastGt)
    {
        int nTicks = SceneParams.ObserveTicksPhase2 - SceneParams.SlideStartObserveTick;
        double dist = SceneParams.SlideSpeed * nTicks;
        var vis = w.Things.Where(t => lastGt.Count(x => x == t.GtId) >= ScoringParams.VisibleMinPx).ToList();
        rng.Shuffle(vis);
        int n = 1 + rng.Integer(2);
        var slides = new List<(Thing, double[], int)>();
        foreach (var t in vis)
        {
            if (slides.Count >= n) break;
            for (int k = 0; k < 40; k++)
            {
                double ang = rng.Uniform(0, 2 * Math.PI);
                var d = new[] { Math.Sin(ang), Math.Cos(ang) };
                var end = new[] { t.Pos[0] + d[0] * dist, t.Pos[1] + d[1] * dist };
                bool ok = Math.Sqrt(end[0] * end[0] + end[1] * end[1]) <= SceneParams.PlacementRadius + 0.15 &&
                          new[] { 0.33, 0.66, 1.0 }.All(f => w.IsFree(new[] { t.Pos[0] + d[0] * dist * f, t.Pos[1] + d[1] * dist * f }, t.FootprintRadius, t));
                if (ok) { slides.Add((t, new[] { d[0] * SceneParams.SlideSpeed, d[1] * SceneParams.SlideSpeed }, nTicks)); break; }
            }
        }
        return slides;
    }
}
