using System.Security.Cryptography;

namespace TinyBrain.Learner;

/// <summary>
/// Epistemic states. Only raw observations are KNOWN (Experiment 1 decision D1). Every inference —
/// including "this is the same thing as before" — is BELIEVED or UNKNOWN.
/// </summary>
public enum Epistemic { Known, Believed, Unknown }

/// <summary>
/// What was actually perceived. Append-only. Each observation is copied on arrival (so the sender
/// cannot change it afterwards) and hashed, so anyone can verify it was never altered.
/// </summary>
public sealed class ObservationStore
{
    private readonly Dictionary<string, Observation> _obs = new();
    public Dictionary<string, string> Hashes { get; } = new();

    public string Add(Observation o)
    {
        string id = $"obs:{o.Tick}";
        var copy = new Observation(o.Tick, (byte[])o.Left.Clone(), (byte[])o.Right.Clone(), o.SelfMotion);
        _obs[id] = copy;
        Hashes[id] = HashOf(copy);
        return id;
    }

    /// <summary>Read-only view: callers get the bytes as ReadOnlySpan so they cannot write into memory.</summary>
    public ReadOnlySpan<byte> Left(string id) => _obs[id].Left;
    public ReadOnlySpan<byte> Right(string id) => _obs[id].Right;
    public SelfMotion SelfMotion(string id) => _obs[id].SelfMotion;
    public int Count => _obs.Count;

    public static string HashOf(Observation o)
    {
        using var sha = SHA256.Create();
        var sm = System.Text.Encoding.UTF8.GetBytes($"{o.SelfMotion.DForward:R}|{o.SelfMotion.DRight:R}|{o.SelfMotion.DYaw:R}");
        sha.TransformBlock(o.Left, 0, o.Left.Length, null, 0);
        sha.TransformBlock(o.Right, 0, o.Right.Length, null, 0);
        sha.TransformFinalBlock(sm, 0, sm.Length);
        return Convert.ToHexString(sha.Hash!);
    }
}

/// <summary>
/// One derived claim. Provenance is mandatory: which mechanism (name@version#params) made it and from
/// which observations / earlier inferences. Records are never edited; a revision is a new record.
/// </summary>
public sealed record Inference(string Id, int Tick, string Kind, string Mechanism, IReadOnlyList<string> Inputs,
                               Epistemic Epistemic, IReadOnlyDictionary<string, object?> Payload);

public sealed class InferenceStore
{
    private readonly List<Inference> _records = new();
    public IReadOnlyList<Inference> Records => _records;

    public Inference Add(int tick, string kind, string mechanism, IEnumerable<string> inputs, Epistemic epistemic,
                         Dictionary<string, object?>? payload = null)
    {
        var ins = inputs.ToList();
        if (epistemic == Epistemic.Known)
            throw new ArgumentException("No inference may be KNOWN: only raw observations are known.");
        if (string.IsNullOrEmpty(mechanism) || ins.Count == 0)
            throw new ArgumentException("Inference without provenance.");
        var rec = new Inference($"inf:{_records.Count + 1}", tick, kind, mechanism, ins, epistemic,
                                payload ?? new Dictionary<string, object?>());
        _records.Add(rec);
        return rec;
    }

    public int MissingProvenance => _records.Count(r => string.IsNullOrEmpty(r.Mechanism) || r.Inputs.Count == 0);
}
