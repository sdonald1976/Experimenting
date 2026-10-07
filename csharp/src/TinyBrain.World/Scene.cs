namespace TinyBrain.World;

/// <summary>Scene design parameters (the world, not the learner). Same values as Experiment 2's Python scene.</summary>
public static class SceneParams
{
    public const double RoomHalfSize = 4.0, PlacementRadius = 1.15, MinGap = 0.2, ArcRadius = 2.3;
    public const double ExploreArcDeg = 40, ExploreStepDeg = 1, TurnStepDeg = 12, ReturnArcDeg = 35, MinMoveDistance = 0.45;
    public const int AwayMoveTicks = 8, ObserveTicksPhase1 = 10, ObserveTicksPhase2 = 30, Phase1Cycles = 4;
    public const double SlideSpeed = 0.03;           // m per tick
    public const int SlideStartObserveTick = 10;     // sliding starts after the 10-tick evaluation look
    public static readonly string[] Phase2Kinds = { "swap", "swap", "move", "novel", "replace", "plain" };

    public static readonly Dictionary<string, double[]> Palette = new()
    {
        ["red"] = new[] { 0.80, 0.22, 0.17 }, ["green"] = new[] { 0.22, 0.68, 0.27 }, ["blue"] = new[] { 0.20, 0.32, 0.85 },
        ["yellow"] = new[] { 0.85, 0.74, 0.16 }, ["purple"] = new[] { 0.58, 0.26, 0.76 }, ["orange"] = new[] { 0.90, 0.50, 0.12 },
        ["cyan"] = new[] { 0.15, 0.68, 0.74 }, ["pink"] = new[] { 0.90, 0.42, 0.62 },
    };
    public static readonly double[] Floor = { 0.5, 0.5, 0.5 };
    public static readonly double[][] Walls = { new[] { 0.78, 0.66, 0.50 }, new[] { 0.52, 0.62, 0.78 }, new[] { 0.62, 0.76, 0.56 }, new[] { 0.80, 0.58, 0.66 } };
    public const string Canary = "CANARY-GT-LEAK-cs-3d91e0";   // must never appear in the learner's state
}

/// <summary>
/// HARNESS-SIDE ground truth: the room and six things with deliberately unequal evidence reliability
/// (three share a colour, two share another; two identical spheres; two identical elongated boxes; ...).
/// </summary>
public sealed class World
{
    public Rng Rng { get; }
    public List<Plane> Structure { get; } = new();
    public List<Thing> Things { get; } = new();
    public Dictionary<int, string> Names { get; } = new() { [1] = "FLOOR", [2] = "WALL_E", [3] = "WALL_W", [4] = "WALL_N", [5] = "WALL_S" };
    private int _nextId = 10;

    public World(int seed)
    {
        var rng = Rng = new Rng(seed);
        double Wd = SceneParams.RoomHalfSize;
        Plane P(int id, double[] pt, double[] n, double[] alb) => new() { GtId = id, Point = pt, Normal = n, Albedo = alb, Texture = new Texture(rng) };
        Structure.Add(P(1, new[] { 0.0, 0, 0 }, new[] { 0.0, 1, 0 }, SceneParams.Floor));
        Structure.Add(P(2, new[] { Wd, 0, 0 }, new[] { -1.0, 0, 0 }, SceneParams.Walls[0]));
        Structure.Add(P(3, new[] { -Wd, 0, 0 }, new[] { 1.0, 0, 0 }, SceneParams.Walls[1]));
        Structure.Add(P(4, new[] { 0.0, 0, Wd }, new[] { 0.0, 0, -1 }, SceneParams.Walls[2]));
        Structure.Add(P(5, new[] { 0.0, 0, -Wd }, new[] { 0.0, 0, 1 }, SceneParams.Walls[3]));

        var cols = SceneParams.Palette.Keys.ToList();
        rng.Shuffle(cols);
        string c1 = cols[0], c2 = cols[1], c3 = cols[2];
        double rS = rng.Uniform(0.11, 0.15);
        var elong = new[] { rng.Uniform(0.20, 0.24), rng.Uniform(0.07, 0.10), rng.Uniform(0.06, 0.08) };
        double cube = rng.Uniform(0.10, 0.13);
        var tall = new[] { rng.Uniform(0.06, 0.08), rng.Uniform(0.38, 0.46), 0 };
        var specs = new (string shape, string col, double[] size)[]
        {
            ("sphere", c1, new[] { rS, 0, 0 }), ("box", c1, elong), ("box", c1, new[] { cube, cube, cube }),
            ("cylinder", c2, tall), ("sphere", c2, new[] { rS, 0, 0 }), ("box", c3, elong),
        };
        for (int i = 0; i < specs.Length; i++)
        {
            var (shape, col, size) = specs[i];
            AddThing(shape, col, (double[])size.Clone(), FreePosition(Footprint(shape, size)), rng.Uniform(0, 2 * Math.PI), $"THING_{"ABCDEF"[i]}");
        }
    }

    public static double Footprint(string shape, double[] size) => shape == "box" ? Math.Sqrt(size[0] * size[0] + size[2] * size[2]) : size[0];

    public static double[] RandomSize(Rng rng, string shape) => shape switch
    {
        "box" => new[] { rng.Uniform(0.10, 0.20), rng.Uniform(0.10, 0.20), rng.Uniform(0.10, 0.20) },
        "cylinder" => new[] { rng.Uniform(0.09, 0.17), rng.Uniform(0.20, 0.45), 0 },
        _ => new[] { rng.Uniform(0.10, 0.18), 0, 0 },
    };

    private double[] SurfacePoint(string shape, double[] size)
    {
        var rng = Rng;
        if (shape == "sphere")
        {
            var u = new[] { rng.Normal(), Math.Abs(rng.Normal()) * 0.6 + 0.1, rng.Normal() };
            double n = V3.Norm(u); return new[] { size[0] * u[0] / n, size[0] * u[1] / n, size[0] * u[2] / n };
        }
        if (shape == "box")
        {
            var u = new[] { rng.Normal(), Math.Abs(rng.Normal()), rng.Normal() };
            double m = Enumerable.Range(0, 3).Max(k => Math.Abs(u[k]) / size[k]);
            return new[] { u[0] / m, u[1] / m, u[2] / m };
        }
        double r = size[0], h = size[1];
        if (rng.Uniform() < 0.25)
        {
            double a = rng.Uniform(0, 2 * Math.PI), rr = r * Math.Sqrt(rng.Uniform());
            return new[] { rr * Math.Cos(a), h / 2, rr * Math.Sin(a) };
        }
        double b = rng.Uniform(0, 2 * Math.PI);
        return new[] { r * Math.Cos(b), rng.Uniform(-h / 2, h / 2), r * Math.Sin(b) };
    }

    public Thing AddThing(string shape, string colour, double[] size, double[] pos, double yaw, string label)
    {
        int nSpots = 3 + Rng.Integer(4);
        var spots = Enumerable.Range(0, nSpots).Select(_ => (SurfacePoint(shape, size), Rng.Uniform(0.022, 0.045))).ToList();
        var t = new Thing
        {
            GtId = _nextId++, Shape = shape, Size = size, Pos = pos, Yaw = yaw, Albedo = SceneParams.Palette[colour],
            Texture = new Texture(Rng), Spots = spots, Name = $"{label}_{colour}_{shape}", Colour = colour,
        };
        Things.Add(t);
        Names[t.GtId] = t.Name;
        return t;
    }

    public bool IsFree(double[] p, double sizeR, Thing? exclude = null) =>
        Things.Where(t => t != exclude).All(t => Math.Sqrt(Math.Pow(t.Pos[0] - p[0], 2) + Math.Pow(t.Pos[1] - p[1], 2)) >= t.FootprintRadius + sizeR + SceneParams.MinGap);

    public double[] FreePosition(double sizeR, Thing? exclude = null, double[]? minFrom = null, int tries = 800)
    {
        for (int i = 0; i < tries; i++)
        {
            double r = SceneParams.PlacementRadius * Math.Sqrt(Rng.Uniform()), a = Rng.Uniform(0, 2 * Math.PI);
            var p = new[] { r * Math.Sin(a), r * Math.Cos(a) };
            if (IsFree(p, sizeR, exclude) && (minFrom is null || Math.Sqrt(Math.Pow(p[0] - minFrom[0], 2) + Math.Pow(p[1] - minFrom[1], 2)) >= SceneParams.MinMoveDistance))
                return p;
        }
        throw new InvalidOperationException("no free position");
    }

    public IReadOnlyList<Surface> Surfaces => Structure.Cast<Surface>().Concat(Things).ToList();

    public static double[] ArcPos(double alpha) => new[] { SceneParams.ArcRadius * Math.Sin(alpha), SceneParams.ArcRadius * Math.Cos(alpha) };

    /// <summary>Viewing direction (deg) of the camera in the thing's own frame. Harness-only bookkeeping for "experienced viewpoint".</summary>
    public static double ObjectAzimuth(Thing t, double[] cam)
    {
        double bearing = Math.Atan2(cam[0] - t.Pos[0], cam[1] - t.Pos[1]);
        double a = (bearing - t.Yaw + Math.PI) % (2 * Math.PI);
        if (a < 0) a += 2 * Math.PI;
        return (a - Math.PI) * 180 / Math.PI;
    }
}

/// <summary>One scripted tick of the passive camera route (harness only).</summary>
public sealed record TickPlan(double[] Pos, double Psi, double Alpha, int Phase, string Part, int? Cycle, string? EventKind, double? ReturnAlpha, bool SlideStart);

public static class Schedule
{
    /// <summary>Exploration arc, 4 static look-away cycles (phase 1), then 6 cycles with events and visible sliding (phase 2).</summary>
    public static List<TickPlan> Build(World world)
    {
        var rng = world.Rng;
        double Rad(double d) => d * Math.PI / 180;
        var ticks = new List<TickPlan>();
        void Add(double a, double psi, int phase, string part, int? cycle = null, string? ev = null, double? ret = null, bool slide = false)
            => ticks.Add(new TickPlan(World.ArcPos(a), psi, a, phase, part, cycle, ev, ret, slide));

        double alpha = 0, E = Rad(SceneParams.ExploreArcDeg), step = Rad(SceneParams.ExploreStepDeg);
        Add(alpha, alpha + Math.PI, 1, "explore");
        while (alpha > -E + 1e-9) { alpha -= step; Add(alpha, alpha + Math.PI, 1, "explore"); }
        while (alpha < E - 1e-9) { alpha += step; Add(alpha, alpha + Math.PI, 1, "explore"); }

        var kinds2 = SceneParams.Phase2Kinds.ToList();
        rng.Shuffle(kinds2);
        var plan = Enumerable.Repeat((1, "plain"), SceneParams.Phase1Cycles).Concat(kinds2.Select(k => (2, k))).ToList();
        double psi = alpha + Math.PI, turn = Rad(SceneParams.TurnStepDeg);
        int nTurn = (int)Math.Round(Math.PI / turn);
        for (int ci = 0; ci < plan.Count; ci++)
        {
            var (phase, kind) = plan[ci];
            for (int i = 0; i < nTurn; i++) { psi += turn; Add(alpha, psi, phase, "turn_away", ci); }
            double target = alpha;
            while (Math.Abs(target - alpha) < Rad(10)) target = rng.Uniform(-Rad(SceneParams.ReturnArcDeg), Rad(SceneParams.ReturnArcDeg));
            double a0 = alpha;
            for (int k = 1; k <= SceneParams.AwayMoveTicks; k++)
            {
                alpha = a0 + (target - a0) * k / SceneParams.AwayMoveTicks;
                Add(alpha, alpha + (psi - a0), phase, "away", ci, k == 1 ? kind : null, k == 1 ? target : null);
            }
            psi = alpha + (psi - a0);
            for (int i = 0; i < nTurn; i++) { psi += turn; Add(alpha, psi, phase, "turn_back", ci); }
            int nObs = phase == 1 ? SceneParams.ObserveTicksPhase1 : SceneParams.ObserveTicksPhase2;
            for (int k = 0; k < nObs; k++) Add(alpha, psi, phase, "observe", ci, slide: phase == 2 && k == SceneParams.SlideStartObserveTick);
        }
        return ticks;
    }
}
