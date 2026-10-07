using System.Text.Json;
using TinyBrain.Learner;
using TinyBrain.World;

// TinyBrain runner: one or more seeds of the Experiment 2 world, several learner variants, a readable summary.
//
//   dotnet run -c Release --project src/TinyBrain.Runner -- --seed 0
//   dotnet run -c Release --project src/TinyBrain.Runner -- --seeds 0-3 --variants A_learned,C_equal --out results.json
//   dotnet run -c Release --project src/TinyBrain.Runner -- --seed 0 --frames frames --frame-every 25
//   dotnet run -c Release --project src/TinyBrain.Runner -- --describe        (print every mechanism and its assumptions)

var opts = ParseArgs(args);
if (opts.ContainsKey("describe"))
{
    foreach (var v in LearnerConfig.AllVariants)
    {
        Console.WriteLine($"=== variant {v}");
        foreach (var m in LearnerConfig.Variant(v).Mechanisms) Console.WriteLine(m.Describe());
    }
    return;
}
var seeds = opts.TryGetValue("seeds", out var sr) ? ParseRange(sr) : new List<int> { int.Parse(opts.GetValueOrDefault("seed", "0")) };
var variants = opts.GetValueOrDefault("variants", string.Join(",", LearnerConfig.AllVariants)).Split(',');
string? framesDir = opts.GetValueOrDefault("frames");
int frameEvery = int.Parse(opts.GetValueOrDefault("frame-every", "25"));
if (framesDir != null) Directory.CreateDirectory(framesDir);

var results = new List<SeedResult>();
foreach (int seed in seeds)
{
    Console.WriteLine($"--- seed {seed}: variants {string.Join(", ", variants)}");
    var res = ExperimentRun.Run(seed, variants, Console.WriteLine,
        framesDir == null ? null : (tick, L, R, gt, ex) =>
        {
            if (tick % frameEvery == 0) Bmp.WriteFrame(Path.Combine(framesDir, $"seed{seed}_t{tick:D4}.bmp"), 256, 192, L, R, ex.Labels);
        });
    results.Add(res);
    PrintSeed(res);
}
if (seeds.Count > 1) PrintPooled(results);
if (opts.TryGetValue("out", out var outPath))
{
    File.WriteAllText(outPath, JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"wrote {outPath}");
}

static void PrintSeed(SeedResult r)
{
    Console.WriteLine($"\nseed {r.Seed}: {r.Ticks} ticks in {r.WallSeconds:F0}s; phase-2 events: {string.Join(", ", r.CycleKinds.Skip(4))}" +
                      (r.SkippedEvents.Count > 0 ? $"; skipped: {string.Join(",", r.SkippedEvents)}" : ""));
    Console.WriteLine("things (ground truth, shown to YOU only): " + string.Join(", ", r.ThingNames.Where(kv => kv.Key >= 10).Select(kv => kv.Value)));
    PrintTable(r.Variants.Values.Select(v => (v.Variant, v.Trials)).ToList());
    foreach (var v in r.Variants.Values)
    {
        Console.WriteLine($"\n[{v.Variant}] units={v.Units} records={v.Records} decisions={string.Join(" ", v.DecisionCounts.Select(kv => $"{kv.Key}:{kv.Value}"))}");
        Console.WriteLine("  audits: " + string.Join("  ", v.Audit.Select(kv => $"{kv.Key}={(kv.Value is List<string> l ? (l.Count == 0 ? "none" : string.Join("|", l)) : kv.Value)}")));
        foreach (var u in v.MappedUnits.OrderByDescending(u => u.Records).Take(4))
            Console.WriteLine($"  unit {u.Unit,4} (~{u.Thing}, {u.Records} records, created t={u.CreatedTick}) learned usefulness: " +
                              string.Join("  ", u.LearnedUsefulness.OrderBy(kv => kv.Key).Select(kv => $"{Short(kv.Key)}={kv.Value:F2}")));
    }
}

static void PrintPooled(List<SeedResult> rs)
{
    Console.WriteLine($"\n=== pooled over {rs.Count} seeds");
    var variants = rs[0].Variants.Keys.ToList();
    PrintTable(variants.Select(v => (v, rs.SelectMany(r => r.Variants[v].Trials).ToList())).ToList());
}

static void PrintTable(List<(string variant, List<Trial> trials)> rows)
{
    var conds = new (string name, Func<Trial, bool> pred)[]
    {
        ("phase1 reappear", t => t.Phase == 1 && t.Type == "unmoved" && Eligible(t)),
        ("phase2 unmoved", t => t.Phase == 2 && t.Type == "unmoved" && Eligible(t)),
        ("swapped", t => t.Type == "swapped" && Eligible(t)),
        ("moved", t => t.Type == "moved" && Eligible(t)),
        ("novel+replaced", t => t.Type is "novel" or "replace_new"),
    };
    string[] outs = { "correct_same", "false_same", "new", "unknown", "not_detected" };
    Console.WriteLine($"\n{"condition",-16} {"variant",-18} {"n",4}  {"correct",8} {"FALSE",8} {"new",8} {"UNKNOWN",8} {"undetect",8}");
    foreach (var (name, pred) in conds)
        foreach (var (variant, trials) in rows)
        {
            var ts = trials.Where(pred).ToList();
            string Pct(string o) => ts.Count == 0 ? "-" : $"{100.0 * ts.Count(t => t.Outcome == o) / ts.Count:F0}%";
            Console.WriteLine($"{name,-16} {variant,-18} {ts.Count,4}  " + string.Join(" ", outs.Select(o => $"{Pct(o),8}")));
        }
}

static bool Eligible(Trial t) => t.ExperiencedViewpoint == true && t.WasUnobserved && t.EventUnobserved;
static string Short(string k) => k switch { "spatial_extent" => "size", "chroma_distribution" => "colour", "metric_local_patterns_highpass" => "pattern", "location_self_frame" => "location", _ => k };

static Dictionary<string, string> ParseArgs(string[] a)
{
    var d = new Dictionary<string, string>();
    for (int i = 0; i < a.Length; i++)
        if (a[i].StartsWith("--")) d[a[i][2..]] = i + 1 < a.Length && !a[i + 1].StartsWith("--") ? a[++i] : "true";
    return d;
}

static List<int> ParseRange(string s)
{
    var p = s.Split('-');
    return p.Length == 2 ? Enumerable.Range(int.Parse(p[0]), int.Parse(p[1]) - int.Parse(p[0]) + 1).ToList() : s.Split(',').Select(int.Parse).ToList();
}

/// <summary>Writes left | right | learner's unit claims (each unit a colour) as a 24-bit BMP.</summary>
static class Bmp
{
    public static void WriteFrame(string path, int w, int h, byte[] L, byte[] R, ushort[] labels)
    {
        int W3 = 3 * w, rowBytes = (W3 * 3 + 3) / 4 * 4;
        using var f = new BinaryWriter(File.Create(path));
        f.Write((byte)'B'); f.Write((byte)'M'); f.Write(54 + rowBytes * h); f.Write(0); f.Write(54);
        f.Write(40); f.Write(W3); f.Write(h); f.Write((short)1); f.Write((short)24); f.Write(0); f.Write(rowBytes * h); f.Write(2835); f.Write(2835); f.Write(0); f.Write(0);
        var row = new byte[rowBytes];
        for (int y = h - 1; y >= 0; y--)
        {
            for (int x = 0; x < W3; x++)
            {
                int panel = x / w, px = y * w + x % w;
                byte r, g, b;
                if (panel == 0) (r, g, b) = (L[3 * px], L[3 * px + 1], L[3 * px + 2]);
                else if (panel == 1) (r, g, b) = (R[3 * px], R[3 * px + 1], R[3 * px + 2]);
                else
                {
                    int u = labels[px];
                    (r, g, b) = u == 0 ? ((byte)0, (byte)0, (byte)0) : ((byte)(u * 97 % 200 + 55), (byte)(u * 57 % 200 + 55), (byte)(u * 31 % 200 + 55));
                }
                row[3 * x] = b; row[3 * x + 1] = g; row[3 * x + 2] = r;
            }
            f.Write(row);
        }
    }
}
