namespace TinyBrain.Learner;

/// <summary>
/// What the learner may know about its own sensors (Experiment 1, Q1): "my eyes are this far apart
/// and have these optical properties". Nothing about the world.
/// </summary>
public sealed record Calibration(
    int Width, int Height,
    double Fx, double Fy, double Cx, double Cy,
    double Baseline,      // metres between the two cameras
    double MountPitch);   // camera tilt relative to the body, radians (negative = looking down)

/// <summary>Noisy estimate of the body's own movement since the previous tick (body frame).</summary>
public sealed record SelfMotion(double DForward, double DRight, double DYaw);

/// <summary>
/// One tick of sensory input. This is EVERYTHING the learner ever receives: two RGB images
/// (row-major, 3 bytes per pixel) and a noisy self-motion estimate. No labels, no depth, no IDs.
/// </summary>
public sealed class Observation
{
    public int Tick { get; }
    public byte[] Left { get; }
    public byte[] Right { get; }
    public SelfMotion SelfMotion { get; }

    public Observation(int tick, byte[] left, byte[] right, SelfMotion selfMotion)
    {
        Tick = tick;
        Left = left;
        Right = right;
        SelfMotion = selfMotion;
    }
}
