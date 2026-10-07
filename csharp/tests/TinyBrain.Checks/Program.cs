using TinyBrain.Learner;

// Architecture checks (no test framework needed):  dotnet run --project tests/TinyBrain.Checks
// Each check prints PASS/FAIL; exit code is the number of failures.

int failures = 0;
void Check(string name, Func<bool> f)
{
    bool ok;
    try { ok = f(); } catch (Exception e) { Console.WriteLine($"  exception: {e.Message}"); ok = false; }
    Console.WriteLine($"{(ok ? "PASS" : "FAIL")}  {name}");
    if (!ok) failures++;
}
bool Throws(Action a) { try { a(); return false; } catch (ArgumentException) { return true; } }

var cal = new Calibration(32, 24, 30, 30, 15.5, 11.5, 0.12, -0.5);

Check("learner assembly does not reference the world/harness assembly", () =>
    typeof(Learner).Assembly.GetReferencedAssemblies().All(a => a.Name != "TinyBrain.World"));

Check("observation that does not match the calibration is rejected", () =>
    Throws(() => new Learner(cal, LearnerConfig.Variant("A_learned")).Step(new Observation(0, new byte[10], new byte[10], new SelfMotion(0, 0, 0)))));

Check("stored observations are copies: changing the sender's array does not change memory", () =>
{
    var store = new ObservationStore();
    var left = new byte[] { 1, 2, 3 };
    var id = store.Add(new Observation(0, left, new byte[] { 4, 5, 6 }, new SelfMotion(0, 0, 0)));
    left[0] = 99;
    return store.Left(id)[0] == 1;
});

Check("no inference may be KNOWN; provenance is mandatory", () =>
{
    var s = new InferenceStore();
    return Throws(() => s.Add(0, "x", "m", new[] { "obs:0" }, Epistemic.Known))
        && Throws(() => s.Add(0, "x", "m", Array.Empty<string>(), Epistemic.Believed))
        && Throws(() => s.Add(0, "x", "", new[] { "obs:0" }, Epistemic.Believed));
});

Unit MakeUnit(int id)
{
    var u = new Unit(id, 0, "NEW", "inf:1");
    u.Records.Add(new ExperienceRecord($"exp:{id}:0", id, 0, "obs:0", "creation", 0,
        new Dictionary<string, object?> { ["a"] = new[] { 0.0 }, ["b"] = new[] { 0.0 } }, "inf:2", 100));
    return u;
}
void Informative(Unit u, string k, int n = 40)
{
    for (int i = 0; i < n; i++) { u.AddSample("W", k, new Sample(0.01, i, "r", "p")); u.AddSample("B", k, new Sample(1.0, i, "q", "s")); }
}

Check("usefulness starts uncommitted (0) and learning is local to one unit", () =>
{
    var U = new WindowedAucUsefulness();
    var u1 = MakeUnit(1); var u2 = MakeUnit(2);
    bool start = U.Usefulness(u1, "a") == 0 && U.Vote(u1, "a", 0.3) == 0;
    Informative(u2, "a");
    return start && U.Usefulness(u2, "a") > 0.5 && U.Usefulness(u1, "a") == 0;
});

Check("recency window lets usefulness be revised, but no sample is deleted", () =>
{
    var U = new WindowedAucUsefulness { WWindow = 10, BWindow = 20 };
    var u = MakeUnit(1);
    Informative(u, "a");
    double high = U.Usefulness(u, "a");
    for (int i = 40; i < 60; i++) u.AddSample("W", "a", new Sample(1.5, i, "r", "p"));
    return U.Usefulness(u, "a") < 0.2 * high && u.W["a"].Count == 60;
});

// A scalar evidence kind for unit checks
var kinds = new Dictionary<string, EvidenceKind> { ["a"] = new Scalar("a"), ["b"] = new Scalar("b") };
var ctx = new RecognitionContext { Kinds = kinds, Usefulness = new WindowedAucUsefulness() };

Check("no single kind is definitive, for every weighting, in both directions", () =>
{
    foreach (var mode in new[] { WeightSource.Learned, WeightSource.Fixed, WeightSource.Equal })
    {
        var rec = new WeightedVote { Weights = mode, FixedWeights = new() { ["a"] = 0.5, ["b"] = 0.5 } };
        var u = MakeUnit(1);
        Informative(u, "a");    // only 'a' carries information
        var same = rec.UnitDecision(new() { new Dictionary<string, object?> { ["a"] = new[] { 0.0 }, ["b"] = new[] { 0.7 } } }, u, ctx).d;
        var diff = rec.UnitDecision(new() { new Dictionary<string, object?> { ["a"] = new[] { 5.0 }, ["b"] = new[] { 0.7 } } }, u, ctx).d;
        if (same == Decision.Same || diff == Decision.New) return false;
    }
    return true;
});

Check("with no learned basis, the learned-weight learner says UNKNOWN", () =>
    new WeightedVote { Weights = WeightSource.Learned }
        .UnitDecision(new() { new Dictionary<string, object?> { ["a"] = new[] { 0.0 }, ["b"] = new[] { 0.0 } } }, MakeUnit(1), ctx).d == Decision.Unknown);

Check("every mechanism declares status and assumptions", () =>
    LearnerConfig.AllVariants.SelectMany(v => LearnerConfig.Variant(v).Mechanisms)
        .All(m => (m.Status is "EXPERIMENTAL" or "BASELINE") && m.Assumptions.Count > 0));

Console.WriteLine(failures == 0 ? "\nall checks passed" : $"\n{failures} check(s) FAILED");
return failures;

sealed class Scalar : EvidenceKind
{
    private readonly string _n;
    public Scalar(string n) => _n = n;
    public override string Name => _n;
    public override IReadOnlyList<string> Assumptions => new[] { "test scalar" };
    public override object? Extract(RegionContext ctx, int[] pix) => null;
    public override double Distance(object a, object b) => Math.Abs(((double[])a)[0] - ((double[])b)[0]);
}
