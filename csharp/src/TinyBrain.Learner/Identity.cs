namespace TinyBrain.Learner;

public enum Decision { Same, New, Unknown }

/// <summary>
/// Experiment 2's usefulness learner. Builds "same" (W) and "different" (B) samples per unit and evidence kind,
/// turns a query distance into a vote, and learns an explicit per-unit usefulness weight.
/// </summary>
public sealed class WindowedAucUsefulness : Mechanism
{
    public int WMinGapTicks { get; init; } = 9;
    public int WWindow { get; init; } = 30;
    public int BWindow { get; init; } = 60;
    public double PseudoCount { get; init; } = 1.0;
    public double HardQuantile { get; init; } = 0.25;
    public double ShrinkN0 { get; init; } = 10.0;

    public override string Slot => "usefulness";
    public override string Name => "windowed_auc";
    public override string Version => "2";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
    {
        ["w_min_gap_ticks"] = WMinGapTicks, ["w_window"] = WWindow, ["b_window"] = BWindow, ["pseudo_count"] = PseudoCount,
        ["hard_quantile"] = HardQuantile, ["shrink_n0"] = ShrinkN0, ["vote"] = "nearest",
    };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "W = best-match distance from a new CONTINUITY record to the unit's earlier records in the same uninterrupted segment that are >= w_min_gap_ticks older.",
        "B = best-match distance from a record of ANOTHER unit observed on the same tick to this unit's earlier records. Simultaneously observed regions are assumed different things.",
        "No recognition- or revision-derived identity ever creates a sample (no feedback path).",
        "Only the most recent w_window W and b_window B samples are used (recency window); all older samples stay stored.",
        "Vote v(d) = (2 P(|d-w| < |d-b|) - 1) * n/(n + pseudo_count): is d closer to past same-thing or past different-thing distances? 0 with no samples.",
        "Usefulness w(u,k) = n/(n+shrink_n0) * max(0, 2 AUC - 1), AUC = P(w < b) between windowed W and the hardest hard_quantile of windowed B. Starts at 0.",
    };

    public Dictionary<string, (double d, string partner)> Within(Unit u, ExperienceRecord rec, IReadOnlyDictionary<string, EvidenceKind> kinds)
    {
        var prior = u.Records.Take(u.Records.Count - 1)
            .Where(x => x.Segment == rec.Segment && rec.Tick - x.Tick >= WMinGapTicks).ToList();
        return BestMatch(rec, prior, kinds);
    }

    public Dictionary<string, (double d, string partner)> Between(Unit u, ExperienceRecord other, IReadOnlyDictionary<string, EvidenceKind> kinds, int tick)
        => BestMatch(other, u.Records.Where(x => x.Tick < tick).ToList(), kinds);

    private static Dictionary<string, (double, string)> BestMatch(ExperienceRecord rec, List<ExperienceRecord> prior, IReadOnlyDictionary<string, EvidenceKind> kinds)
    {
        var outp = new Dictionary<string, (double, string)>();
        foreach (var (k, ex) in kinds)
        {
            if (rec.Evidence[k] is not { } a) continue;
            double best = double.MaxValue; string? id = null;
            foreach (var x in prior)
            {
                if (x.Evidence[k] is not { } b) continue;
                double d = ex.Distance(a, b);
                if (d < best) { best = d; id = x.Id; }
            }
            if (id != null) outp[k] = (best, id);
        }
        return outp;
    }

    private (double[] W, double[] B) Window(Unit u, string k)
    {
        double[] Take(Dictionary<string, List<Sample>> d, int n) =>
            d.TryGetValue(k, out var l) ? l.Skip(Math.Max(0, l.Count - n)).Select(s => s.Distance).ToArray() : Array.Empty<double>();
        return (Take(u.W, WWindow), Take(u.B, BWindow));
    }

    public double Vote(Unit u, string k, double d)
    {
        var (W, B) = Window(u, k);
        if (W.Length == 0 || B.Length == 0) return 0;
        double wins = 0;
        foreach (var w in W)
            foreach (var b in B)
            {
                double dw = Math.Abs(d - w), db = Math.Abs(d - b);
                wins += dw < db ? 1 : dw == db ? 0.5 : 0;
            }
        double p = wins / (W.Length * B.Length);
        int n = Math.Min(W.Length, B.Length);
        return (2 * p - 1) * n / (n + PseudoCount);
    }

    public double Usefulness(Unit u, string k)
    {
        var (W, B) = Window(u, k);
        if (W.Length == 0 || B.Length == 0) return 0;
        var hard = B.OrderBy(v => v).Take(Math.Max(1, (int)Math.Ceiling(HardQuantile * B.Length))).ToArray();
        double wins = 0;
        foreach (var w in W)
            foreach (var b in hard)
                wins += w < b ? 1 : w == b ? 0.5 : 0;
        double auc = wins / (W.Length * hard.Length);
        int n = Math.Min(W.Length, B.Length);
        return n / (n + ShrinkN0) * Math.Max(0, 2 * auc - 1);
    }
}

public sealed record RecognitionResult(Decision Decision, int? Group, int? Unit, List<int> Possible, string Detail);

public sealed class RecognitionContext
{
    public required IReadOnlyDictionary<string, EvidenceKind> Kinds { get; init; }
    public required WindowedAucUsefulness Usefulness { get; init; }
}

/// <summary>Recognition: compare a query (one or more experience records' evidence) with existing identity groups.</summary>
public abstract class Recognition : Mechanism
{
    public override string Slot => "recognition";
    public abstract RecognitionResult Decide(List<IReadOnlyDictionary<string, object?>> queries, List<(int group, List<Unit> units)> groups, RecognitionContext ctx);

    /// <summary>Smallest distance between any query and any of the unit's records, per kind ("best-matching experience").</summary>
    protected static Dictionary<string, double> MinDistances(IReadOnlyDictionary<string, object?> q, Unit u, IEnumerable<string> kinds, RecognitionContext ctx)
    {
        var outp = new Dictionary<string, double>();
        foreach (var k in kinds)
        {
            if (q[k] is not { } a) continue;
            double best = double.MaxValue;
            foreach (var r in u.Records)
                if (r.Evidence[k] is { } b) best = Math.Min(best, ctx.Kinds[k].Distance(a, b));
            if (best < double.MaxValue) outp[k] = best;
        }
        return outp;
    }
}

public enum WeightSource { Learned, Fixed, Equal }

/// <summary>
/// The shared decision rule of Experiment 2. Variants differ ONLY in where weights come from.
/// S = sum(w*v)/sum(w); SAME iff S >= threshold and still >= threshold with any one kind removed (no single
/// kind defines identity); NEW mirrors at -threshold; otherwise UNKNOWN.
/// </summary>
public sealed class WeightedVote : Recognition
{
    public double Threshold { get; init; } = 0.5;
    public WeightSource Weights { get; init; } = WeightSource.Learned;
    public double MinMass { get; init; } = 0.5;
    public Dictionary<string, double> FixedWeights { get; init; } = new()
    {
        ["metric_local_patterns_highpass"] = 0.35, ["chroma_distribution"] = 0.30, ["spatial_extent"] = 0.20, ["location_self_frame"] = 0.15,
    };

    public override string Name => "weighted_vote";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
        { ["threshold"] = Threshold, ["weights"] = Weights.ToString(), ["min_mass"] = MinMass };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Per kind: d = best-match distance between the query and the unit's records; vote v = usefulness.Vote(d).",
        "Weights: Learned = the unit's own learned usefulness; Fixed = a predetermined guess (same for all units, never changes); Equal = 1.",
        "SAME iff S >= threshold and S >= threshold with ANY one kind removed; NEW mirrors at -threshold; else UNKNOWN. Same rule for every variant.",
        "Learned only: total weight < min_mass -> UNKNOWN (no learned basis yet).",
        "Several query records (revision): each scored separately, per-kind median vote.",
        "Across groups: exactly one SAME -> SAME; several -> UNKNOWN; none SAME but some UNKNOWN -> UNKNOWN; all NEW -> NEW.",
    };

    public (Decision d, double S) UnitDecision(List<IReadOnlyDictionary<string, object?>> queries, Unit u, RecognitionContext ctx)
    {
        var votesPer = new Dictionary<string, List<double>>();
        foreach (var q in queries)
            foreach (var (k, d) in MinDistances(q, u, ctx.Kinds.Keys, ctx))
            {
                if (!votesPer.TryGetValue(k, out var l)) votesPer[k] = l = new List<double>();
                l.Add(ctx.Usefulness.Vote(u, k, d));
            }
        var votes = votesPer.ToDictionary(kv => kv.Key, kv => Median(kv.Value));
        var w = votes.Keys.ToDictionary(k => k, k => Weights switch
        {
            WeightSource.Learned => ctx.Usefulness.Usefulness(u, k),
            WeightSource.Fixed => FixedWeights[k],
            _ => 1.0,
        });
        double Sof(IEnumerable<string> ks)
        {
            double m = ks.Sum(k => w[k]);
            return m > 0 ? ks.Sum(k => w[k] * votes[k]) / m : 0;
        }
        var all = votes.Keys.ToList();
        double S = Sof(all);
        var loo = all.Select(k => Sof(all.Where(x => x != k))).ToList();
        if (all.Count < 2 || (Weights == WeightSource.Learned && w.Values.Sum() < MinMass)) return (Decision.Unknown, S);
        if (S >= Threshold && loo.Min() >= Threshold) return (Decision.Same, S);
        if (S <= -Threshold && loo.Max() <= -Threshold) return (Decision.New, S);
        return (Decision.Unknown, S);
    }

    private static double Median(List<double> v)
    {
        var a = v.OrderBy(x => x).ToArray();
        return a.Length % 2 == 1 ? a[a.Length / 2] : 0.5 * (a[a.Length / 2 - 1] + a[a.Length / 2]);
    }

    public override RecognitionResult Decide(List<IReadOnlyDictionary<string, object?>> queries, List<(int group, List<Unit> units)> groups, RecognitionContext ctx)
    {
        var per = new List<(int group, Decision d, double S, int unit)>();
        foreach (var (g, units) in groups)
        {
            (int rank, Decision d, double S, int unit)? best = null;
            foreach (var u in units)
            {
                var (d, S) = UnitDecision(queries, u, ctx);
                int rank = d == Decision.Same ? 2 : d == Decision.Unknown ? 1 : 0;
                if (best is null || rank > best.Value.rank || (rank == best.Value.rank && S > best.Value.S)) best = (rank, d, S, u.Id);
            }
            per.Add((g, best!.Value.d, best.Value.S, best.Value.unit));
        }
        string detail = string.Join("; ", per.OrderByDescending(p => p.S).Take(3).Select(p => $"g{p.group}:{p.d}:{p.S:F2}"));
        var same = per.Where(p => p.d == Decision.Same).OrderByDescending(p => p.S).ToList();
        var unk = per.Where(p => p.d == Decision.Unknown).OrderByDescending(p => p.S).ToList();
        if (same.Count == 1) return new(Decision.Same, same[0].group, same[0].unit, new(), detail);
        if (same.Count > 1) return new(Decision.Unknown, null, null, same.Select(p => p.group).ToList(), detail);
        if (unk.Count > 0) return new(Decision.Unknown, null, null, unk.Select(p => p.group).ToList(), detail);
        return new(Decision.New, null, null, new(), detail);
    }
}

/// <summary>Shortcut baseline: always SAME as the group whose recorded location is nearest.</summary>
public sealed class LocationOnly : Recognition
{
    public override string Name => "baseline_nearest_location";
    public override string Status => "BASELINE";
    public override IReadOnlyList<string> Assumptions => new[] { "Always SAME as the group whose recorded location is nearest (location-only shortcut)." };

    public override RecognitionResult Decide(List<IReadOnlyDictionary<string, object?>> queries, List<(int group, List<Unit> units)> groups, RecognitionContext ctx)
    {
        (double d, int g, int u)? best = null;
        foreach (var (g, units) in groups)
            foreach (var u in units)
                if (MinDistances(queries[0], u, new[] { "location_self_frame" }, ctx).TryGetValue("location_self_frame", out var d)
                    && (best is null || d < best.Value.d)) best = (d, g, u.Id);
        return best is null ? new(Decision.New, null, null, new(), "")
                            : new(Decision.Same, best.Value.g, best.Value.u, new(), $"nearest d={best.Value.d:F2}");
    }
}

/// <summary>Shortcut baseline: never links anything.</summary>
public sealed class AlwaysNew : Recognition
{
    public override string Name => "baseline_always_new";
    public override string Status => "BASELINE";
    public override IReadOnlyList<string> Assumptions => new[] { "Every non-continuous candidate is declared a new thing." };
    public override RecognitionResult Decide(List<IReadOnlyDictionary<string, object?>> queries, List<(int group, List<Unit> units)> groups, RecognitionContext ctx)
        => new(Decision.New, null, null, new(), "");
}

/// <summary>Identity revision: a young NEW/PROVISIONAL unit is re-compared (all its records) with older groups.</summary>
public sealed class YoungUnitRevision : Mechanism
{
    public bool Enabled { get; init; } = true;
    public int MaxRecords { get; init; } = 12;
    public override string Slot => "revision";
    public override string Name => Enabled ? "young_unit_reevaluation" : "no_revision";
    public override string Status => Enabled ? "EXPERIMENTAL" : "BASELINE";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object> { ["max_records"] = MaxRecords };
    public override IReadOnlyList<string> Assumptions => Enabled
        ? new[]
        {
            "A unit created as NEW or PROVISIONAL and not yet linked to an older identity is re-compared, using all its records, against older groups each time it gains a continuity record, until it has max_records records.",
            "SAME -> a new same_identity inference (both units remain; nothing is erased). Changed possibly_same_as -> a new inference.",
            "Groups ever observed on the same tick as the unit are excluded (believed distinct).",
        }
        : new[] { "Identity decisions are never revisited." };
}
