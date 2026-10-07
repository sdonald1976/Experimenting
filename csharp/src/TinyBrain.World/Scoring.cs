using TinyBrain.Learner;

namespace TinyBrain.World;

/// <summary>Scoring thresholds (harness parameters, same as Experiments 1-2).</summary>
public static class ScoringParams
{
    public const int ClaimMinOverlapPx = 30, VisibleMinPx = 150, AssocMinTotalPx = 200;
    public const double ClaimMinPurity = 0.5, AssocMinPurity = 0.5, ViewpointToleranceDeg = 15;
}

/// <summary>
/// HARNESS-SIDE: compares one learner variant's exports with ground truth. Nothing here is ever sent back.
/// Outcomes: correct_same, false_same, new, unknown (never an error), not_detected.
/// </summary>
public sealed class VariantScorer
{
    private readonly Dictionary<int, Dictionary<int, int>> _counts = new();   // unit -> gt id -> pixels claimed so far
    private readonly Dictionary<int, int> _parent = new();                    // learner's believed identities (from its exports)

    public int Find(int x)
    {
        if (!_parent.ContainsKey(x)) _parent[x] = x;
        while (_parent[x] != x) { _parent[x] = _parent[_parent[x]]; x = _parent[x]; }
        return x;
    }

    public void Update(LearnerExport ex, int[] gt)
    {
        foreach (var e in ex.IdentityEvents)
            if (e.Kind == "same" && e.Other is int o) _parent[Find(e.Unit)] = Find(o);
        for (int i = 0; i < gt.Length; i++)
        {
            int u = ex.Labels[i];
            if (u == 0) continue;
            if (!_counts.TryGetValue(u, out var d)) _counts[u] = d = new();
            d[gt[i]] = d.GetValueOrDefault(gt[i]) + 1;
        }
    }

    /// <summary>Which physical thing each unit has mostly claimed so far (taken before a world change).</summary>
    public Dictionary<int, int> AssociationSnapshot()
    {
        var snap = new Dictionary<int, int>();
        foreach (var (u, d) in _counts)
        {
            int tot = d.Values.Sum();
            var best = d.MaxBy(kv => kv.Value);
            if (tot >= ScoringParams.AssocMinTotalPx && best.Value / (double)tot >= ScoringParams.AssocMinPurity) snap[u] = best.Key;
        }
        return snap;
    }

    /// <summary>The unit that claims thing g on this tick, or null.</summary>
    public static int? ClaimedUnit(LearnerExport ex, int[] gt, int g)
    {
        var overlap = new Dictionary<int, int>(); var total = new Dictionary<int, int>();
        for (int i = 0; i < gt.Length; i++)
        {
            int u = ex.Labels[i];
            if (u == 0) continue;
            total[u] = total.GetValueOrDefault(u) + 1;
            if (gt[i] == g) overlap[u] = overlap.GetValueOrDefault(u) + 1;
        }
        int? best = null; int bo = 0;
        foreach (var (u, o) in overlap)
            if (o >= ScoringParams.ClaimMinOverlapPx && o / (double)total[u] >= ScoringParams.ClaimMinPurity && o > bo) { best = u; bo = o; }
        return best;
    }

    public string Outcome(LearnerExport ex, int[] gt, int g, Dictionary<int, int> snapshot)
    {
        var p = ClaimedUnit(ex, gt, g);
        if (p is null) return "not_detected";
        var claim = ex.Claims[p.Value];
        int root = Find(p.Value);
        var members = _counts.Keys.Where(u => Find(u) == root).DefaultIfEmpty(p.Value).ToList();
        bool priorG = members.Any(u => snapshot.TryGetValue(u, out var x) && x == g);
        bool priorOther = members.Any(u => snapshot.TryGetValue(u, out var x) && x != g);
        if (claim.Epistemic == Epistemic.Unknown) return "unknown";
        if (priorOther) return "false_same";
        return priorG ? "correct_same" : "new";
    }
}
