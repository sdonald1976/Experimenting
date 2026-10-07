namespace TinyBrain.Learner;

/// <summary>Small, dependency-free image helpers (the equivalents of the scipy calls in the Python version).</summary>
public static class ImageOps
{
    /// <summary>RGB bytes (row-major, 3 per pixel) to grey = mean of the three channels.</summary>
    public static float[] ToGray(ReadOnlySpan<byte> rgb, int w, int h)
    {
        var g = new float[w * h];
        for (int i = 0; i < w * h; i++)
            g[i] = (rgb[3 * i] + rgb[3 * i + 1] + rgb[3 * i + 2]) / 3f;
        return g;
    }

    /// <summary>Mean over a size x size window; image edges are extended (scipy mode="nearest").</summary>
    public static float[] BoxFilter(float[] src, int w, int h, int size)
    {
        int r = size / 2;
        var tmp = new float[w * h];
        var dst = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float s = 0;
                for (int k = -r; k <= r; k++) s += src[y * w + Math.Clamp(x + k, 0, w - 1)];
                tmp[y * w + x] = s / size;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float s = 0;
                for (int k = -r; k <= r; k++) s += tmp[Math.Clamp(y + k, 0, h - 1) * w + x];
                dst[y * w + x] = s / size;
            }
        return dst;
    }

    /// <summary>Separable Gaussian blur (kernel truncated at 4 sigma), edges extended.</summary>
    public static float[] Gaussian(float[] src, int w, int h, double sigma)
    {
        if (sigma <= 0) return (float[])src.Clone();
        int r = Math.Max(1, (int)Math.Ceiling(4 * sigma));
        var k = new double[2 * r + 1];
        double sum = 0;
        for (int i = -r; i <= r; i++) { k[i + r] = Math.Exp(-0.5 * i * i / (sigma * sigma)); sum += k[i + r]; }
        for (int i = 0; i < k.Length; i++) k[i] /= sum;
        var tmp = new float[w * h];
        var dst = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0;
                for (int i = -r; i <= r; i++) s += k[i + r] * src[y * w + Math.Clamp(x + i, 0, w - 1)];
                tmp[y * w + x] = (float)s;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0;
                for (int i = -r; i <= r; i++) s += k[i + r] * tmp[Math.Clamp(y + i, 0, h - 1) * w + x];
                dst[y * w + x] = (float)s;
            }
        return dst;
    }

    /// <summary>Bilinear sample at a real-valued position; 0 outside the image.</summary>
    public static double Bilinear(float[] img, int w, int h, double y, double x)
    {
        int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y);
        double fx = x - x0, fy = y - y0;
        double V(int yy, int xx) => (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0 : img[yy * w + xx];
        return (1 - fy) * ((1 - fx) * V(y0, x0) + fx * V(y0, x0 + 1)) + fy * ((1 - fx) * V(y0 + 1, x0) + fx * V(y0 + 1, x0 + 1));
    }
}
