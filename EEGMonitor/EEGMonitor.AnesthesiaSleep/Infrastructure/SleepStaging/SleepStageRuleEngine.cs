using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;

namespace EEGMonitor.AnesthesiaSleep.Infrastructure.SleepStaging;

public class SleepStageRuleEngine
{
    private const int EpochDurationSeconds = 30;
    private const int MinEpochsForTransition = 3;

    private readonly List<ProcessedEEGResult> _epochBuffer = new(32);
    private int _epochIndex;

    public SleepEpoch ClassifyEpoch(List<ProcessedEEGResult> secondChunks)
    {
        if (secondChunks.Count == 0)
            return CreateUnknownEpoch();

        var avg = ComputeAverages(secondChunks);
        var epoch = new SleepEpoch
        {
            EpochIndex = _epochIndex++,
            StartTime = secondChunks[0].Timestamp,
            EndTime = secondChunks[^1].Timestamp,
            DeltaPowerAvg = avg.Delta,
            ThetaPowerAvg = avg.Theta,
            AlphaPowerAvg = avg.Alpha,
            BetaPowerAvg = avg.Beta,
            GammaPowerAvg = avg.Gamma,
            SpindleDensityAvg = avg.SpindleDensity,
            HeartRateAvg = avg.HeartRate,
            SpO2Avg = avg.SpO2,
            SeValueAvg = avg.SE,
            ReSeDiffAvg = avg.ReSeDiff
        };

        if (avg.IsHighArtifact)
        {
            epoch.Stage = SleepStage.Movement;
            epoch.IsArtifact = true;
            epoch.Confidence = 0.9;
        }
        else
        {
            (epoch.Stage, epoch.Confidence) = Classify(avg);
        }

        return epoch;
    }

    public SleepStage SmoothStage(SleepStage current, IReadOnlyList<SleepStage> previousWindow)
    {
        if (previousWindow.Count < MinEpochsForTransition)
            return current;

        var lastN = previousWindow.TakeLast(MinEpochsForTransition).ToList();
        if (lastN.All(s => s == current))
            return current;

        var majority = previousWindow.TakeLast(5)
            .Where(s => s != SleepStage.Movement && s != SleepStage.Unknown)
            .GroupBy(s => s)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault();

        return majority?.Key ?? current;
    }

    private static SleepEpoch CreateUnknownEpoch()
    {
        return new SleepEpoch { Stage = SleepStage.Unknown, Confidence = 0, IsArtifact = true };
    }

    private static EpochAverages ComputeAverages(List<ProcessedEEGResult> chunks)
    {
        var n = chunks.Count;
        double d = 0, t = 0, a = 0, b = 0, g = 0, sp = 0, hr = 0, spo2 = 0, se = 0, reSe = 0;
        int artifactCount = 0, hrCount = 0, spo2Count = 0, seCount = 0;

        foreach (var c in chunks)
        {
            d += c.DeltaPower;
            t += c.ThetaPower;
            a += c.AlphaPower;
            b += c.BetaPower;
            g += c.GammaPower;
            sp += c.SpindleDensity;

            if (c.HeartRate.HasValue) { hr += c.HeartRate.Value; hrCount++; }
            if (c.SpO2.HasValue) { spo2 += c.SpO2.Value; spo2Count++; }
            if (c.StateEntropy.HasValue) { se += c.StateEntropy.Value; seCount++; }
            if (c.StateEntropy.HasValue && c.ResponseEntropy.HasValue)
                reSe += c.ResponseEntropy.Value - c.StateEntropy.Value;

            if (c.EegAmplitudeUv < 0.5 || c.EegTonalRatio > 0.4 || c.HwIsSaturated)
                artifactCount++;
        }

        return new EpochAverages
        {
            Delta = d / n,
            Theta = t / n,
            Alpha = a / n,
            Beta = b / n,
            Gamma = g / n,
            SpindleDensity = sp / n,
            HeartRate = hrCount > 0 ? hr / hrCount : 0,
            SpO2 = spo2Count > 0 ? spo2 / spo2Count : 0,
            SE = seCount > 0 ? se / seCount : 0,
            ReSeDiff = seCount > 0 ? reSe / seCount : 0,
            IsHighArtifact = (double)artifactCount / n > 0.5
        };
    }

    private static (SleepStage, double) Classify(EpochAverages avg)
    {
        double emg = avg.ReSeDiff > 0 ? avg.ReSeDiff : 0;
        double slowRatio = avg.Delta;
        double spindleDensity = avg.SpindleDensity;

        // Wake: high beta/EMG, low delta, high dominant frequency proxy (beta/(theta+alpha))
        double arousalScore = avg.Beta / (avg.Theta + avg.Alpha + 0.001);
        if (emg > 8 && arousalScore > 0.4 && slowRatio < 0.35)
            return (SleepStage.Wake, ClampConfidence(0.7 + arousalScore * 0.3));

        // N3: delta-dominant, low spindle, low EMG
        if (slowRatio > 0.42 && emg < 5 && spindleDensity < 2.0)
            return (SleepStage.N3, ClampConfidence(0.6 + slowRatio * 0.4));

        // N2: spindles present, moderate delta, moderate EMG
        if (spindleDensity >= 2.0 && emg < 8 && slowRatio < 0.45)
            return (SleepStage.N2, ClampConfidence(0.55 + spindleDensity * 0.1));

        // REM: very low EMG, theta/alpha mixed, low delta
        if (emg < 3 && slowRatio < 0.30 && avg.SE > 20)
            return (SleepStage.REM, ClampConfidence(0.55 + (1.0 - emg / 10.0) * 0.3));

        // N1: low delta, low spindle, moderate EMG, theta present
        if (slowRatio < 0.38 && spindleDensity < 2.0 && emg < 10)
            return (SleepStage.N1, ClampConfidence(0.4 + avg.Theta * 0.3));

        // Fallback: most likely N2 if spindles present, otherwise Wake
        if (spindleDensity >= 1.0)
            return (SleepStage.N2, 0.45);
        return (SleepStage.Wake, 0.4);
    }

    private static double ClampConfidence(double v) => Math.Clamp(v, 0.0, 1.0);

    private sealed class EpochAverages
    {
        public double Delta, Theta, Alpha, Beta, Gamma;
        public double SpindleDensity, HeartRate, SpO2, SE, ReSeDiff;
        public bool IsHighArtifact;
    }
}
