namespace TinyBrain.Learner;

/// <summary>
/// Every mechanism the architecture does NOT itself require is a Mechanism: it states its status
/// (EXPERIMENTAL or BASELINE), version, parameters and assumptions in plain words. Its Tag is written
/// into every inference it produces, so any belief can be traced to the exact rule that made it.
/// Swapping a mechanism = constructing a different subclass in LearnerConfig.
/// </summary>
public abstract class Mechanism
{
    public abstract string Slot { get; }
    public abstract string Name { get; }
    public virtual string Version => "1";
    public virtual string Status => "EXPERIMENTAL";
    public abstract IReadOnlyList<string> Assumptions { get; }

    /// <summary>Parameter values, for logging. Subclasses expose them as typed properties too.</summary>
    public virtual IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>();

    public string Tag => $"{Slot}:{Name}@{Version}";

    public string Describe()
    {
        var p = string.Join(", ", Parameters.Select(kv => $"{kv.Key}={kv.Value}"));
        var a = string.Join("\n", Assumptions.Select(x => "    - " + x));
        return $"{Tag} [{Status}] ({p})\n{a}";
    }
}
