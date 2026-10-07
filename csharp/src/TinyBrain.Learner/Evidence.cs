namespace TinyBrain.Learner;

/// <summary>Everything an evidence extractor may look at for one candidate region on one tick.</summary>
public sealed class RegionContext
{
    public required double[][] WorldPoints { get; init; }   // learner's self-motion frame, NaN where no depth
    public required double[][] CamPoints { get; init; }
    public required float[] ChromaR { get; init; }
    public required float[] ChromaG { get; init; }
    public required float[] Bright { get; init; }
    public required float[] Gray { get; init; }
    public required Calibration Calib { get; init; }
}

/// <summary>
/// A semantically neutral evidence kind (Q6a). Names are for humans; the learner treats each only as
/// "evidence kind k". Extract returns null when the region gives no usable value.
/// </summary>
public abstract class EvidenceKind : Mechanism
{
    public override string Slot => "evidence";
    public abstract object? Extract(RegionContext ctx, int[] pix);
    public abstract double Distance(object a, object b);

    protected static List<double[]> FinitePoints(double[][] pts, int[] pix) =>
        pix.Select(i => pts[i]).Where(p => !double.IsNaN(p[0]) && !double.IsNaN(p[1]) && !double.IsNaN(p[2])).ToList();

    protected static double Median(IEnumerable<double> xs)
    {
        var a = xs.OrderBy(v => v).ToArray();
        return a.Length % 2 == 1 ? a[a.Length / 2] : 0.5 * (a[a.Length / 2 - 1] + a[a.Length / 2]);
    }

    /// <summary>Percentile with linear interpolation (numpy default).</summary>
    protected static double Percentile(double[] sorted, double q)
    {
        double pos = q / 100.0 * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos), hi = Math.Min(lo + 1, sorted.Length - 1);
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }
}

/// <summary>Metric size of the visible surface: sorted spreads along principal axes (rotation-invariant).</summary>
public sealed class SpatialExtent : EvidenceKind
{
    public override string Name => "spatial_extent";
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Metric size of the VISIBLE surface: sorted 5-95% spreads of the region's 3-D points along their principal axes. Needs the allowed calibration.",
        "Distance = mean |log ratio| of the three sorted spreads.",
    };

    public override object? Extract(RegionContext ctx, int[] pix)
    {
        var P = FinitePoints(ctx.WorldPoints, pix);
        if (P.Count < 30) return null;
        var med = Enumerable.Range(0, 3).Select(k => Median(P.Select(p => p[k]))).ToArray();
        var C = P.Select(p => new[] { p[0] - med[0], p[1] - med[1], p[2] - med[2] }).ToList();
        var mean = Enumerable.Range(0, 3).Select(k => C.Average(p => p[k])).ToArray();
        var cov = new double[3, 3];
        foreach (var p in C)
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    cov[a, b] += (p[a] - mean[a]) * (p[b] - mean[b]);
        var axes = LinearAlgebra.SymmetricEigenvectors3(cov);
        var spreads = new double[3];
        for (int k = 0; k < 3; k++)
        {
            var proj = C.Select(p => p[0] * axes[k][0] + p[1] * axes[k][1] + p[2] * axes[k][2]).OrderBy(v => v).ToArray();
            spreads[k] = Percentile(proj, 95) - Percentile(proj, 5);
        }
        return spreads.OrderByDescending(v => v).Select(v => v + 1e-3).ToArray();
    }

    public override double Distance(object a, object b)
    {
        var x = (double[])a; var y = (double[])b;
        return Enumerable.Range(0, 3).Average(k => Math.Abs(Math.Log(x[k] / y[k])));
    }
}

/// <summary>Distribution of pixel chromaticity (brightness-invariant colour).</summary>
public sealed class ChromaDistribution : EvidenceKind
{
    public int Bins { get; init; } = 16;
    public override string Name => "chroma_distribution";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object> { ["bins"] = Bins, ["min_brightness"] = 30.0 };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "2-D histogram of pixel chromaticity (r, g fractions) over the region; brightness-invariant.",
        "Distance = Hellinger distance between histograms.",
    };

    public override object? Extract(RegionContext ctx, int[] pix)
    {
        var h = new double[Bins * Bins];
        int n = 0;
        foreach (int i in pix)
        {
            if (ctx.Bright[i] <= 30.0) continue;
            int br = Math.Min(Bins - 1, (int)(ctx.ChromaR[i] * Bins));
            int bg = Math.Min(Bins - 1, (int)(ctx.ChromaG[i] * Bins));
            h[br * Bins + bg]++; n++;
        }
        if (n < 30) return null;
        return h.Select(v => Math.Sqrt(v / n)).ToArray();
    }

    public override double Distance(object a, object b)
    {
        var x = (double[])a; var y = (double[])b;
        double s = 0;
        for (int i = 0; i < x.Length; i++) s += (x[i] - y[i]) * (x[i] - y[i]);
        return Math.Sqrt(s) / Math.Sqrt(2);
    }
}

/// <summary>
/// Stable local visual patterns at a fixed PHYSICAL patch size (scale normalised by stereo depth), after
/// removing smooth shading (high-pass). Detects no category: any stable pattern counts.
/// </summary>
public sealed class LocalPatternsHighPass : EvidenceKind
{
    public double PatchM { get; init; } = 0.05;
    public int Samples { get; init; } = 9;
    public int MaxPatches { get; init; } = 24;
    public double MinStd { get; init; } = 6.0;
    public double HighpassSigmaFrac { get; init; } = 0.35;

    public override string Name => "metric_local_patterns_highpass";
    public override string Version => "2";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
        { ["patch_m"] = PatchM, ["samples"] = Samples, ["max_patches"] = MaxPatches, ["min_std"] = MinStd, ["highpass_sigma_frac"] = HighpassSigmaFrac };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Grey-level patches of fixed physical size (patch_m), resampled to samples x samples using stereo depth.",
        "Smooth shading is removed first (subtract a Gaussian blur with sigma = highpass_sigma_frac * patch size).",
        "Patches lie fully inside the region; the most varied are kept; each is zero-mean, unit-norm.",
        "Distance = 1 - symmetric mean best normalised cross-correlation between the two patch sets.",
    };

    public override object? Extract(RegionContext ctx, int[] pix)
    {
        var c = ctx.Calib; int W = c.Width, H = c.Height;
        var zs = pix.Select(i => ctx.CamPoints[i][2]).Where(z => !double.IsNaN(z)).ToList();
        if (zs.Count < 30) return null;
        double sizePx = c.Fx * PatchM / Median(zs);
        if (sizePx < 4) return null;
        int ymin = pix.Min(i => i / W), ymax = pix.Max(i => i / W), xmin = pix.Min(i => i % W), xmax = pix.Max(i => i % W);
        // high-pass within a margin around the region
        int m = (int)(3 * sizePx) + 2;
        int y0 = Math.Max(0, ymin - m), y1 = Math.Min(H, ymax + m + 1), x0 = Math.Max(0, xmin - m), x1 = Math.Min(W, xmax + m + 1);
        int cw = x1 - x0, chh = y1 - y0;
        var crop = new float[cw * chh];
        for (int y = 0; y < chh; y++) for (int x = 0; x < cw; x++) crop[y * cw + x] = ctx.Gray[(y + y0) * W + x + x0];
        var blur = ImageOps.Gaussian(crop, cw, chh, HighpassSigmaFrac * sizePx);
        var gray = (float[])ctx.Gray.Clone();
        for (int y = 0; y < chh; y++) for (int x = 0; x < cw; x++) gray[(y + y0) * W + x + x0] = crop[y * cw + x] - blur[y * cw + x];

        var mask = new bool[W * H];
        foreach (int i in pix) mask[i] = true;
        int stride = Math.Max(2, (int)(sizePx / 2));
        var offs = Enumerable.Range(0, Samples).Select(k => (k - (Samples - 1) / 2.0) * sizePx / (Samples - 1)).ToArray();
        var patches = new List<(double std, double[] v)>();
        for (int cy = ymin; cy <= ymax; cy += stride)
            for (int cx = xmin; cx <= xmax; cx += stride)
            {
                if (!mask[cy * W + cx]) continue;
                var v = new double[Samples * Samples];
                bool ok = true;
                for (int a = 0; a < Samples && ok; a++)
                    for (int b = 0; b < Samples && ok; b++)
                    {
                        double sy = cy + offs[a], sx = cx + offs[b];
                        int ri = (int)Math.Round(sy), rj = (int)Math.Round(sx);
                        if (ri < 0 || ri >= H || rj < 0 || rj >= W || !mask[ri * W + rj]) { ok = false; break; }
                        v[a * Samples + b] = ImageOps.Bilinear(gray, W, H, sy, sx);
                    }
                if (!ok) continue;
                double mean = v.Average();
                double std = Math.Sqrt(v.Average(x => (x - mean) * (x - mean)));
                patches.Add((std, v));
            }
        var kept = patches.OrderByDescending(p => p.std).Take(MaxPatches).Where(p => p.std >= MinStd).ToList();
        if (kept.Count == 0) return null;
        return kept.Select(p =>
        {
            double mean = p.v.Average();
            var z = p.v.Select(x => x - mean).ToArray();
            double norm = Math.Sqrt(z.Sum(x => x * x));
            return z.Select(x => x / norm).ToArray();
        }).ToArray();
    }

    public override double Distance(object a, object b)
    {
        var A = (double[][])a; var B = (double[][])b;
        var rowMax = new double[A.Length]; var colMax = Enumerable.Repeat(double.MinValue, B.Length).ToArray();
        for (int i = 0; i < A.Length; i++)
        {
            rowMax[i] = double.MinValue;
            for (int j = 0; j < B.Length; j++)
            {
                double s = 0;
                for (int k = 0; k < A[i].Length; k++) s += A[i][k] * B[j][k];
                if (s > rowMax[i]) rowMax[i] = s;
                if (s > colMax[j]) colMax[j] = s;
            }
        }
        return 1 - 0.5 * (rowMax.Average() + colMax.Average());
    }
}

/// <summary>Where the region is, in the learner's own dead-reckoned frame (drifts). Evidence, never definitive.</summary>
public sealed class LocationSelfFrame : EvidenceKind
{
    public override string Name => "location_self_frame";
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Median 3-D point of the region in the learner's own dead-reckoned frame (drifts).",
        "Distance = Euclidean. Location is evidence, never definitive.",
    };

    public override object? Extract(RegionContext ctx, int[] pix)
    {
        var P = FinitePoints(ctx.WorldPoints, pix);
        if (P.Count < 30) return null;
        return Enumerable.Range(0, 3).Select(k => Median(P.Select(p => p[k]))).ToArray();
    }

    public override double Distance(object a, object b)
    {
        var x = (double[])a; var y = (double[])b;
        return Math.Sqrt(Enumerable.Range(0, 3).Sum(k => (x[k] - y[k]) * (x[k] - y[k])));
    }
}

public static class LinearAlgebra
{
    /// <summary>Eigenvectors of a symmetric 3x3 matrix (Jacobi rotations). Returned as rows.</summary>
    public static double[][] SymmetricEigenvectors3(double[,] a0)
    {
        var a = (double[,])a0.Clone();
        var v = new double[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = Math.Abs(a[0, 1]) + Math.Abs(a[0, 2]) + Math.Abs(a[1, 2]);
            if (off < 1e-14) break;
            for (int p = 0; p < 2; p++)
                for (int q = p + 1; q < 3; q++)
                {
                    if (Math.Abs(a[p, q]) < 1e-18) continue;
                    double theta = (a[q, q] - a[p, p]) / (2 * a[p, q]);
                    double t = Math.Sign(theta == 0 ? 1 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < 3; k++)
                    {
                        double akp = a[k, p], akq = a[k, q];
                        a[k, p] = c * akp - s * akq; a[k, q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double apk = a[p, k], aqk = a[q, k];
                        a[p, k] = c * apk - s * aqk; a[q, k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double vkp = v[k, p], vkq = v[k, q];
                        v[k, p] = c * vkp - s * vkq; v[k, q] = s * vkp + c * vkq;
                    }
                }
        }
        return Enumerable.Range(0, 3).Select(j => new[] { v[0, j], v[1, j], v[2, j] }).ToArray();
    }
}
