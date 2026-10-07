namespace TinyBrain.Learner;

/// <summary>Integrates the noisy self-motion estimates. The learner's "world frame" is simply its own starting pose.</summary>
public sealed class DeadReckoning : Mechanism
{
    public override string Slot => "self_motion";
    public override string Name => "dead_reckoning";
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Integrates the noisy body-frame self-motion estimates exactly as given; no visual correction, so the pose drifts.",
        "The learner's world frame is its own starting pose (it never learns true world coordinates).",
    };

    public double X { get; private set; }
    public double Z { get; private set; }
    public double Psi { get; private set; }

    public void Step(SelfMotion sm)
    {
        double rx = Math.Cos(Psi), rz = -Math.Sin(Psi);   // body right
        double fx = Math.Sin(Psi), fz = Math.Cos(Psi);    // body forward
        X += sm.DRight * rx + sm.DForward * fx;
        Z += sm.DRight * rz + sm.DForward * fz;
        Psi += sm.DYaw;
    }
}

/// <summary>Hand-designed stereo: sum of absolute differences, winner-takes-all, uniqueness + left-right check.</summary>
public sealed class SadBlockMatching : Mechanism
{
    public int Window { get; init; } = 7;
    public int MaxDisp { get; init; } = 48;
    public double MinDisp { get; init; } = 1.0;
    public double Uniqueness { get; init; } = 0.92;
    public double LrTolerance { get; init; } = 1.0;

    public override string Slot => "stereo";
    public override string Name => "sad_block_matching";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
        { ["window"] = Window, ["max_disp"] = MaxDisp, ["min_disp"] = MinDisp, ["uniqueness"] = Uniqueness, ["lr_tolerance"] = LrTolerance };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Images are rectified (uses the allowed sensor calibration).",
        "Sum-of-absolute-differences on grey levels over a square window; winner-takes-all.",
        "A disparity is kept only if it is unique (best/second-best cost) and left-right consistent; otherwise 'no depth' (not guessed).",
    };

    /// <summary>Returns per-pixel disparity and a validity mask.</summary>
    public (float[] disp, bool[] valid) Compute(float[] L, float[] R, int w, int h)
    {
        const float Big = 1e9f;
        int D = MaxDisp;
        var cost = new float[D][];
        for (int d = 0; d < D; d++)
        {
            int ww = w - d;
            var diff = new float[ww * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < ww; x++)
                    diff[y * ww + x] = Math.Abs(L[y * w + x + d] - R[y * w + x]);
            var f = ImageOps.BoxFilter(diff, ww, h, Window);
            var c = new float[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    c[y * w + x] = x >= d ? f[y * ww + x - d] : Big;
            cost[d] = c;
        }
        var disp = new float[w * h];
        var valid = new bool[w * h];
        var dl = new int[w * h];
        var dr = new int[w * h];
        for (int i = 0; i < w * h; i++)
        {
            int best = 0;
            for (int d = 1; d < D; d++) if (cost[d][i] < cost[best][i]) best = d;
            dl[i] = best;
        }
        // right-view winners from the same cost volume: cost_r[d][x] = cost[d][x + d]
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int best = 0;
                float bv = float.MaxValue;
                for (int d = 0; d < D; d++)
                {
                    float v = x + d < w ? cost[d][y * w + x + d] : Big;
                    if (v < bv) { bv = v; best = d; }
                }
                dr[y * w + x] = best;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x, d0 = dl[i];
                float best = cost[d0][i], second = float.MaxValue;
                for (int d = 0; d < D; d++)
                    if (Math.Abs(d - d0) > 1 && cost[d][i] < second) second = cost[d][i];
                bool unique = best < Uniqueness * second;
                int xs = x - d0;
                bool lr = xs >= 0 && Math.Abs(d0 - dr[y * w + xs]) <= LrTolerance;
                double off = 0;
                if (d0 > 0 && d0 < D - 1)
                {
                    float c0 = cost[d0 - 1][i], c2 = cost[d0 + 1][i];
                    double den = c0 - 2 * best + c2;
                    if (den > 1e-6) off = Math.Clamp(0.5 * (c0 - c2) / den, -0.5, 0.5);
                }
                disp[i] = (float)(d0 + off);
                valid[i] = unique && lr && disp[i] >= MinDisp && best < Big / 2;
            }
        return (disp, valid);
    }
}

/// <summary>Geometry helpers: disparity -> camera-frame points -> the learner's own self-motion frame.</summary>
public static class Geometry
{
    public static double[][] CameraPoints(float[] disp, bool[] valid, Calibration c)
    {
        var pts = new double[disp.Length][];
        for (int i = 0; i < disp.Length; i++)
        {
            if (!valid[i]) { pts[i] = new[] { double.NaN, double.NaN, double.NaN }; continue; }
            int u = i % c.Width, v = i / c.Width;
            double Z = c.Fx * c.Baseline / disp[i];
            pts[i] = new[] { (u - c.Cx) * Z / c.Fx, (v - c.Cy) * Z / c.Fy, Z };
        }
        return pts;
    }

    public static double[][] ToSelfFrame(double[][] cam, Calibration c, DeadReckoning pose)
    {
        double ph = c.MountPitch, cs = Math.Cos(pose.Psi), sn = Math.Sin(pose.Psi);
        var outp = new double[cam.Length][];
        for (int i = 0; i < cam.Length; i++)
        {
            double X = cam[i][0], Y = cam[i][1], Z = cam[i][2];
            double bx = X, by = -Math.Cos(ph) * Y + Math.Sin(ph) * Z, bz = Math.Sin(ph) * Y + Math.Cos(ph) * Z;
            outp[i] = new[] { pose.X + bx * cs + bz * sn, by, pose.Z - bx * sn + bz * cs };
        }
        return outp;
    }
}

/// <summary>"Distinguishable candidate" (Q4): connected pixels with valid depth, not split by a depth or chromaticity jump.</summary>
public sealed class DepthChromaCandidates : Mechanism
{
    public double DispJumpPx { get; init; } = 1.0;
    public double ChromaJump { get; init; } = 0.05;
    public double ChromaMinBrightness { get; init; } = 30.0;
    public int Blur { get; init; } = 3;
    public int MinPixels { get; init; } = 150;

    public override string Slot => "candidates";
    public override string Name => "depth_chroma_boundaries";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
        { ["disp_jump_px"] = DispJumpPx, ["chroma_jump"] = ChromaJump, ["chroma_min_brightness"] = ChromaMinBrightness, ["blur"] = Blur, ["min_pixels"] = MinPixels };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "A candidate = a 4-connected set of pixels with valid depth, not separated by a disparity jump or a chromaticity jump, with at least min_pixels.",
        "Chromaticity (r/(r+g+b), g/(r+g+b)) is used instead of brightness so shading and intensity marks do not split a surface (hand-designed prior).",
        "No notion of object, background or ground: floor and wall pieces are candidates too.",
    };

    /// <summary>Returns a label image (0 = none, 1..n = candidates), plus chromaticity and brightness for evidence extraction.</summary>
    public (int[] labels, int n, float[] chromaR, float[] chromaG, float[] bright) Extract(ReadOnlySpan<byte> rgb, float[] disp, bool[] valid, int w, int h)
    {
        var ch = new float[3][];
        for (int k = 0; k < 3; k++)
        {
            var c = new float[w * h];
            for (int i = 0; i < w * h; i++) c[i] = rgb[3 * i + k];
            ch[k] = ImageOps.BoxFilter(c, w, h, Blur);
        }
        var cr = new float[w * h]; var cg = new float[w * h]; var br = new float[w * h];
        for (int i = 0; i < w * h; i++)
        {
            float s = ch[0][i] + ch[1][i] + ch[2][i] + 1e-6f;
            cr[i] = ch[0][i] / s; cg[i] = ch[1][i] / s; br[i] = s / 3;
        }
        bool Cut(int a, int b)
        {
            if (!valid[a] || !valid[b]) return true;
            if (Math.Abs(disp[a] - disp[b]) > DispJumpPx) return true;
            double dc = Math.Sqrt((cr[a] - cr[b]) * (cr[a] - cr[b]) + (cg[a] - cg[b]) * (cg[a] - cg[b]));
            return dc > ChromaJump && br[a] > ChromaMinBrightness && br[b] > ChromaMinBrightness;
        }
        var boundary = new bool[w * h];
        for (int i = 0; i < w * h; i++) boundary[i] = !valid[i];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                if (x + 1 < w && Cut(i, i + 1)) { boundary[i] = true; boundary[i + 1] = true; }
                if (y + 1 < h && Cut(i, i + w)) { boundary[i] = true; boundary[i + w] = true; }
            }
        // 4-connected labelling of interior pixels, in raster order
        var lab = new int[w * h];
        var sizes = new List<int> { 0 };
        var stack = new Stack<int>();
        int next = 0;
        for (int s = 0; s < w * h; s++)
        {
            if (boundary[s] || lab[s] != 0) continue;
            next++; int count = 0;
            lab[s] = next; stack.Push(s);
            while (stack.Count > 0)
            {
                int p = stack.Pop(); count++;
                int px = p % w, py = p / w;
                foreach (int q in new[] { px > 0 ? p - 1 : -1, px < w - 1 ? p + 1 : -1, py > 0 ? p - w : -1, py < h - 1 ? p + w : -1 })
                    if (q >= 0 && !boundary[q] && lab[q] == 0) { lab[q] = next; stack.Push(q); }
            }
            sizes.Add(count);
        }
        var remap = new int[next + 1];
        int n = 0;
        for (int l = 1; l <= next; l++) if (sizes[l] >= MinPixels) remap[l] = ++n;
        for (int i = 0; i < w * h; i++) lab[i] = remap[lab[i]];
        return (lab, n, cr, cg, br);
    }
}

/// <summary>Continuity: a candidate continues a unit if it overlaps the unit's previous-tick region. Any gap breaks it (D3).</summary>
public sealed class OverlapContinuity : Mechanism
{
    public double MinIou { get; init; } = 0.3;
    public bool CompensateRotation { get; init; } = true;

    public override string Slot => "continuity";
    public override string Name => "overlap_continuity";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object>
        { ["min_iou"] = MinIou, ["compensate_rotation"] = CompensateRotation };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "Uninterrupted continuity = the candidate overlaps (IoU >= min_iou) the region the unit occupied on the immediately preceding tick. ANY one-tick gap breaks continuity.",
        "Rotation compensation: the previous region is shifted horizontally by -fx * d_yaw using the noisy self-motion estimate.",
        "One-to-one greedy assignment by IoU.",
    };

    /// <summary>prev: (unit id, pixel indices on the previous tick). Returns candidate label -> (unit id, IoU).</summary>
    public Dictionary<int, (int unit, double iou)> Link(List<(int unit, int[] pix)> prev, int[] labels, int n, double dYaw, Calibration c)
    {
        int w = c.Width;
        var curSize = new int[n + 1];
        foreach (var l in labels) curSize[l]++;
        int shift = CompensateRotation ? (int)Math.Round(-c.Fx * dYaw) : 0;
        var pairs = new List<(double iou, int unit, int cand)>();
        foreach (var (unit, pix) in prev)
        {
            var inter = new int[n + 1];
            foreach (int p in pix)
            {
                int x = p % w + shift, y = p / w;
                if (x < 0 || x >= w) continue;
                inter[labels[y * w + x]]++;
            }
            for (int cand = 1; cand <= n; cand++)
            {
                if (inter[cand] == 0) continue;
                double iou = inter[cand] / (double)(pix.Length + curSize[cand] - inter[cand]);
                if (iou >= MinIou) pairs.Add((iou, unit, cand));
            }
        }
        var outp = new Dictionary<int, (int, double)>();
        var usedUnits = new HashSet<int>();
        foreach (var (iou, unit, cand) in pairs.OrderByDescending(p => p.iou).ThenByDescending(p => p.unit).ThenByDescending(p => p.cand))
        {
            if (usedUnits.Contains(unit) || outp.ContainsKey(cand)) continue;
            usedUnits.Add(unit);
            outp[cand] = (unit, iou);
        }
        return outp;
    }
}

/// <summary>When experience records are made (Q7).</summary>
public sealed class FixedIntervalSampling : Mechanism
{
    public int Interval { get; init; } = 3;
    public override string Slot => "sampling";
    public override string Name => "fixed_interval";
    public override IReadOnlyDictionary<string, object> Parameters => new Dictionary<string, object> { ["interval"] = Interval };
    public override IReadOnlyList<string> Assumptions => new[]
    {
        "An experience record is made for every active unit on ticks where tick % interval == 0, and immediately when a unit is created or re-linked by recognition.",
        "Records are never merged or averaged.",
    };
    public bool Due(int tick) => tick % Interval == 0;
}
