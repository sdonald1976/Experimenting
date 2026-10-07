namespace TinyBrain.Learner;

/// <summary>
/// One particular encounter with a persistent unit (Q7, Q21): addressable, never merged or averaged.
/// Link says how it joined the unit: "creation", "continuity" (uninterrupted tracking) or
/// "recognition" (an inference after a gap — never used to train usefulness).
/// </summary>
public sealed record ExperienceRecord(string Id, int Unit, int Tick, string ObsId, string Link, int Segment,
                                      IReadOnlyDictionary<string, object?> Evidence, string InfId, int Npix);

/// <summary>One usefulness sample: a distance, when it was observed, and which records produced it.</summary>
public readonly record struct Sample(double Distance, int Tick, string RecordId, string PartnerId);

/// <summary>One logged change of a unit's learned usefulness for an evidence kind, with what caused it.</summary>
public sealed record UsefulnessChange(int Tick, string Kind, double Old, double New, int NW, int NB,
                                      string CauseSet, double CauseDistance, string CauseRecord, string CausePartner);

/// <summary>
/// A persistent unit: an identity created on first distinguishable encounter (Q4). It has no meaning
/// attached — it is "a distinct thing experienced here, then". Everything it accumulates is local to it.
/// </summary>
public sealed class Unit
{
    public int Id { get; }
    public int CreatedTick { get; }
    public string Decision { get; }                  // "NEW" or "PROVISIONAL" (possibly_same_as something)
    public string CreationInf { get; }
    public List<ExperienceRecord> Records { get; } = new();
    public int Segment { get; set; }                 // current uninterrupted-continuity segment
    public Dictionary<string, List<Sample>> W { get; } = new();   // "same thing" samples (continuity only)
    public Dictionary<string, List<Sample>> B { get; } = new();   // "different thing" samples (co-observation)
    public Dictionary<string, double> Useful { get; } = new();    // current learned usefulness per kind
    public List<UsefulnessChange> History { get; } = new();      // every change, append-only
    public HashSet<int> CoVisible { get; } = new();               // units seen at the same moment (believed distinct)
    public List<int> Possible { get; set; } = new();              // current possibly_same_as candidates
    public bool Open { get; set; }                   // provisional identity not yet resolved -> claims are UNKNOWN
    public bool LinkedToOlder { get; set; }
    public int LastSeen { get; set; }
    public int[] LastPix { get; set; } = Array.Empty<int>();

    public Unit(int id, int tick, string decision, string creationInf)
    {
        Id = id; CreatedTick = tick; Decision = decision; CreationInf = creationInf;
        LastSeen = tick; Open = decision == "PROVISIONAL";
    }

    /// <summary>Append-only: samples are never removed or overwritten.</summary>
    public void AddSample(string set, string kind, Sample s)
    {
        var d = set == "W" ? W : B;
        if (!d.TryGetValue(kind, out var list)) d[kind] = list = new List<Sample>();
        list.Add(s);
    }
}
