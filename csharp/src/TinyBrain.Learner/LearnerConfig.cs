namespace TinyBrain.Learner;

/// <summary>
/// Which mechanism fills each slot. This is the experimenter's choice and contains nothing from ground truth.
/// Experiment 2's variants differ ONLY in the recognition weights (A/B/C) or replace recognition by a shortcut (D1/D2).
/// </summary>
public sealed class LearnerConfig
{
    public required string Name { get; init; }
    public DeadReckoning SelfMotion { get; init; } = new();
    public SadBlockMatching Stereo { get; init; } = new();
    public DepthChromaCandidates Candidates { get; init; } = new();
    public OverlapContinuity Continuity { get; init; } = new();
    public FixedIntervalSampling Sampling { get; init; } = new();
    public List<EvidenceKind> Evidence { get; init; } = new()
        { new SpatialExtent(), new ChromaDistribution(), new LocalPatternsHighPass(), new LocationSelfFrame() };
    public WindowedAucUsefulness Usefulness { get; init; } = new();
    public required Recognition Recognition { get; init; }
    public YoungUnitRevision Revision { get; init; } = new();

    public IEnumerable<Mechanism> Mechanisms =>
        new Mechanism[] { SelfMotion, Stereo, Candidates, Continuity, Sampling }.Concat(Evidence)
            .Concat(new Mechanism[] { Usefulness, Recognition, Revision });

    /// <summary>Fresh instances every call: mechanisms such as dead reckoning hold state and must not be shared.</summary>
    public static LearnerConfig Variant(string name) => name switch
    {
        "A_learned" => new() { Name = name, Recognition = new WeightedVote { Weights = WeightSource.Learned } },
        "B_fixed" => new() { Name = name, Recognition = new WeightedVote { Weights = WeightSource.Fixed } },
        "C_equal" => new() { Name = name, Recognition = new WeightedVote { Weights = WeightSource.Equal } },
        "D1_location_only" => new() { Name = name, Recognition = new LocationOnly(), Revision = new YoungUnitRevision { Enabled = false } },
        "D2_always_new" => new() { Name = name, Recognition = new AlwaysNew(), Revision = new YoungUnitRevision { Enabled = false } },
        _ => throw new ArgumentException($"unknown variant {name}"),
    };

    public static readonly string[] AllVariants = { "A_learned", "B_fixed", "C_equal", "D1_location_only", "D2_always_new" };
}
