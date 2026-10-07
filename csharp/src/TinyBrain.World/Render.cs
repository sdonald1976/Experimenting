namespace TinyBrain.World;

/// <summary>Seeded random numbers (uniform, normal, choice) used everywhere in the world.</summary>
public sealed class Rng
{
    private readonly Random _r;
    public Rng(int seed) => _r = new Random(seed);
    public double Uniform(double a = 0, double b = 1) => a + (b - a) * _r.NextDouble();
    public int Integer(int n) => _r.Next(n);
    public double Normal(double mean = 0, double sd = 1)
    {
        double u1 = 1.0 - _r.NextDouble(), u2 = _r.NextDouble();
        return mean + sd * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }
    public T Choice<T>(IReadOnlyList<T> xs) => xs[_r.Next(xs.Count)];
    public void Shuffle<T>(IList<T> xs)
    {
        for (int i = xs.Count - 1; i > 0; i--) { int j = _r.Next(i + 1); (xs[i], xs[j]) = (xs[j], xs[i]); }
    }
}

public static class V3
{
    public static double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    public static double[] Sub(double[] a, double[] b) => new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
    public static double[] Add(double[] a, double[] b, double s = 1) => new[] { a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2] };
    public static double Norm(double[] a) => Math.Sqrt(Dot(a, a));
    /// <summary>Rotate about +y by yaw (same sense as the body yaw).</summary>
    public static double[] Yaw(double[] v, double yaw)
    {
        double c = Math.Cos(yaw), s = Math.Sin(yaw);
        return new[] { c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2] };
    }
}

/// <summary>Fine surface texture: a few random sine waves in the surface's own coordinates (gives stereo something to match).</summary>
public sealed class Texture
{
    private readonly double[][] _k; private readonly double[] _phase; private readonly double _amp;
    public Texture(Rng rng, double amp = 0.22, int n = 5, double kmin = 50, double kmax = 170)
    {
        _amp = amp; _k = new double[n][]; _phase = new double[n];
        for (int i = 0; i < n; i++)
        {
            var d = new[] { rng.Normal(), rng.Normal(), rng.Normal() };
            double len = V3.Norm(d), k = rng.Uniform(kmin, kmax);
            _k[i] = new[] { d[0] / len * k, d[1] / len * k, d[2] / len * k };
            _phase[i] = rng.Uniform(0, 2 * Math.PI);
        }
    }
    public double Eval(double[] local) => 1 + _amp * _k.Select((k, i) => Math.Sin(V3.Dot(local, k) + _phase[i])).Average();
}

/// <summary>A rendered surface belonging to one physical thing (GtId is ground truth: harness only).</summary>
public abstract class Surface
{
    public int GtId { get; init; }
    public required double[] Albedo { get; init; }
    public required Texture Texture { get; init; }
    public List<(double[] center, double radius)> Spots { get; init; } = new();
    public abstract double Intersect(double[] o, double[] d);
    public abstract (double[] normal, double[] local) NormalAndLocal(double[] p);
}

public sealed class Plane : Surface
{
    public required double[] Point { get; init; }
    public required double[] Normal { get; init; }
    public override double Intersect(double[] o, double[] d)
    {
        double den = V3.Dot(d, Normal);
        if (den > -1e-9) return double.PositiveInfinity;
        double t = V3.Dot(V3.Sub(Point, o), Normal) / den;
        return t > 1e-6 ? t : double.PositiveInfinity;
    }
    public override (double[], double[]) NormalAndLocal(double[] p) => (Normal, p);
}

/// <summary>A rigid thing resting on the floor: box (half sizes), vertical cylinder (radius, height) or sphere (radius).</summary>
public sealed class Thing : Surface
{
    public required string Shape { get; init; }
    public required double[] Size { get; init; }
    public double[] Pos { get; set; } = new double[2];       // (x, z) on the floor
    public double Yaw { get; set; }
    public required string Name { get; init; }               // human-readable: logs only, never sent to the learner
    public required string Colour { get; init; }

    public double[] Center => new[] { Pos[0], Shape == "box" ? Size[1] : Shape == "cylinder" ? Size[1] / 2 : Size[0], Pos[1] };
    public double FootprintRadius => Shape == "box" ? Math.Sqrt(Size[0] * Size[0] + Size[2] * Size[2]) : Size[0];

    public override double Intersect(double[] o, double[] d)
    {
        var c = Center;
        if (Shape == "sphere")
        {
            var oc = V3.Sub(o, c);
            double b = V3.Dot(d, oc), cc = V3.Dot(oc, oc) - Size[0] * Size[0], disc = b * b - cc;
            if (disc <= 0) return double.PositiveInfinity;
            double t = -b - Math.Sqrt(disc);
            return t > 1e-6 ? t : double.PositiveInfinity;
        }
        var ol = V3.Yaw(V3.Sub(o, c), -Yaw);
        var dl = V3.Yaw(d, -Yaw);
        if (Shape == "box")
        {
            double tmin = double.NegativeInfinity, tmax = double.PositiveInfinity;
            for (int k = 0; k < 3; k++)
            {
                if (Math.Abs(dl[k]) < 1e-12) { if (Math.Abs(ol[k]) > Size[k]) return double.PositiveInfinity; continue; }
                double t1 = (-Size[k] - ol[k]) / dl[k], t2 = (Size[k] - ol[k]) / dl[k];
                tmin = Math.Max(tmin, Math.Min(t1, t2)); tmax = Math.Min(tmax, Math.Max(t1, t2));
            }
            return tmax >= tmin && tmin > 1e-6 ? tmin : double.PositiveInfinity;
        }
        double r = Size[0], hgt = Size[1], best = double.PositiveInfinity;
        double a = dl[0] * dl[0] + dl[2] * dl[2], bb = 2 * (ol[0] * dl[0] + ol[2] * dl[2]), c2 = ol[0] * ol[0] + ol[2] * ol[2] - r * r;
        double disc2 = bb * bb - 4 * a * c2;
        if (disc2 > 0 && a > 1e-12)
        {
            double ts = (-bb - Math.Sqrt(disc2)) / (2 * a), ys = ol[1] + ts * dl[1];
            if (ts > 1e-6 && Math.Abs(ys) <= hgt / 2) best = ts;
        }
        if (Math.Abs(dl[1]) > 1e-12)
        {
            double tc = (hgt / 2 - ol[1]) / dl[1], px = ol[0] + tc * dl[0], pz = ol[2] + tc * dl[2];
            if (tc > 1e-6 && px * px + pz * pz <= r * r) best = Math.Min(best, tc);
        }
        return best;
    }

    public double[] ToLocal(double[] p) => V3.Yaw(V3.Sub(p, Center), -Yaw);

    public override (double[], double[]) NormalAndLocal(double[] p)
    {
        var loc = ToLocal(p);
        double[] nl;
        if (Shape == "sphere") { double n = V3.Norm(loc); nl = new[] { loc[0] / n, loc[1] / n, loc[2] / n }; }
        else if (Shape == "box")
        {
            int k = 0; double best = -1;
            for (int i = 0; i < 3; i++) { double q = Math.Abs(loc[i]) / Size[i]; if (q > best) { best = q; k = i; } }
            nl = new double[3]; nl[k] = Math.Sign(loc[k]);
        }
        else nl = loc[1] > Size[1] / 2 - 1e-4 ? new[] { 0.0, 1, 0 } : new[] { loc[0] / Size[0], 0, loc[2] / Size[0] };
        return (V3.Yaw(nl, Yaw), loc);
    }
}

/// <summary>
/// Two rectified pinhole cameras on a body. World: y up, floor y = 0. Body yaw psi: forward = (sin psi, 0, cos psi).
/// Camera frame: x right, y down, z forward, pitched by `pitch` relative to the body; right camera `baseline` metres to the right.
/// </summary>
public sealed class Renderer
{
    public int W { get; } public int H { get; }
    public double F { get; } public double Cx { get; } public double Cy { get; }
    public double Baseline { get; } public double Pitch { get; } public double CamHeight { get; }
    public double NoiseSigma { get; }
    private readonly double[] _light;
    private const double Ambient = 0.42;

    public Renderer(int w, int h, double hfovDeg, double baseline, double pitch, double camHeight, double noiseSigma)
    {
        W = w; H = h; F = (w / 2.0) / Math.Tan(hfovDeg * Math.PI / 360);
        Cx = (w - 1) / 2.0; Cy = (h - 1) / 2.0; Baseline = baseline; Pitch = pitch; CamHeight = camHeight; NoiseSigma = noiseSigma;
        var l = new[] { 0.35, 1.0, 0.25 }; double n = V3.Norm(l);
        _light = new[] { l[0] / n, l[1] / n, l[2] / n };
    }

    /// <summary>What a real system may know about its own sensors.</summary>
    public TinyBrain.Learner.Calibration Calibration => new(W, H, F, F, Cx, Cy, Baseline, Pitch);

    public static (double[] right, double[] down, double[] fwd) CameraAxes(double psi, double pitch)
    {
        var right = new[] { Math.Cos(psi), 0, -Math.Sin(psi) };
        var up = new[] { 0.0, 1, 0 };
        var fwd = new[] { Math.Sin(psi), 0, Math.Cos(psi) };
        var camFwd = V3.Add(new[] { up[0] * Math.Sin(pitch), up[1] * Math.Sin(pitch), up[2] * Math.Sin(pitch) }, fwd, Math.Cos(pitch));
        var camDown = V3.Add(new[] { -up[0] * Math.Cos(pitch), -up[1] * Math.Cos(pitch), -up[2] * Math.Cos(pitch) }, fwd, Math.Sin(pitch));
        return (right, camDown, camFwd);
    }

    /// <summary>Renders one camera: linear RGB in [0,1], ground-truth id per pixel (-1 = none).</summary>
    public (double[] rgb, int[] gt) View(IReadOnlyList<Surface> surfaces, double[] origin, double psi)
    {
        var (r, dn, fw) = CameraAxes(psi, Pitch);
        var rgb = new double[W * H * 3];
        var gt = new int[W * H];
        for (int v = 0; v < H; v++)
            for (int u = 0; u < W; u++)
            {
                double xc = (u - Cx) / F, yc = (v - Cy) / F;
                var d = new[] { xc * r[0] + yc * dn[0] + fw[0], xc * r[1] + yc * dn[1] + fw[1], xc * r[2] + yc * dn[2] + fw[2] };
                double len = V3.Norm(d); d[0] /= len; d[1] /= len; d[2] /= len;
                double bt = double.PositiveInfinity; Surface? bs = null;
                foreach (var s in surfaces) { double t = s.Intersect(origin, d); if (t < bt) { bt = t; bs = s; } }
                int i = v * W + u;
                if (bs is null) { gt[i] = -1; continue; }
                var p = V3.Add(origin, d, bt);
                var (nrm, loc) = bs.NormalAndLocal(p);
                double shade = Ambient + (1 - Ambient) * Math.Max(0, V3.Dot(nrm, _light));
                double tex = bs.Texture.Eval(loc);
                foreach (var (sc, sr) in bs.Spots) if (V3.Norm(V3.Sub(loc, sc)) < sr) tex *= 0.42;
                for (int k = 0; k < 3; k++) rgb[3 * i + k] = bs.Albedo[k] * tex * shade;
                gt[i] = bs.GtId;
            }
        return (rgb, gt);
    }

    /// <summary>Left and right uint8 images (the ONLY things sent to the learner) plus left ground-truth ids (harness only).</summary>
    public (byte[] left, byte[] right, int[] gt) Stereo(IReadOnlyList<Surface> surfaces, double[] posXZ, double psi, Rng noise)
    {
        var right = new[] { Math.Cos(psi), 0, -Math.Sin(psi) };
        var oL = new[] { posXZ[0], CamHeight, posXZ[1] };
        var oR = V3.Add(oL, right, Baseline);
        var (l, gt) = View(surfaces, oL, psi);
        var (rr, _) = View(surfaces, oR, psi);
        byte[] ToBytes(double[] im) => im.Select(x => (byte)Math.Clamp(Math.Round(x * 230 + noise.Normal(0, NoiseSigma)), 0, 255)).ToArray();
        return (ToBytes(l), ToBytes(rr), gt);
    }
}
