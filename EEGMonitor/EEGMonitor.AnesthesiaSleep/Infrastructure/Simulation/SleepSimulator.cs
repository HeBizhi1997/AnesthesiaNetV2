using EEGMonitor.AnesthesiaSleep.Models;
using EEGMonitor.Models;

namespace EEGMonitor.AnesthesiaSleep.Infrastructure.Simulation;

public enum SleepScenarioType
{
    NaturalSleep,
    DISE,
    OSAScreening,
    ConsciousSedation
}

public class SleepSimulator
{
    private readonly object _lock = new();
    private CancellationTokenSource? _cts;
    private Task? _runTask;
    private DateTime _sessionStart;
    private int _elapsedSeconds;
    private double _playbackSpeed = 1.0;
    private SleepScenarioType _scenario = SleepScenarioType.NaturalSleep;

    // Scenario state
    private double _drugEffectLevel;
    private int _desatCount;
    private int _sleepOnsetMinute;
    private readonly Random _rng = new(42);

    public bool IsRunning { get; private set; }
    public TimeSpan Elapsed => TimeSpan.FromSeconds(_elapsedSeconds);
    public double PlaybackSpeed
    {
        get => _playbackSpeed;
        set { lock (_lock) _playbackSpeed = Math.Clamp(value, 0.5, 60.0); }
    }

    public event Action<ProcessedEEGResult>? ResultGenerated;
    public event Action<VitalSignsSnapshot>? VitalSignsUpdated;
    public event Action<ClinicalEvent>? ClinicalEventGenerated;
    public event Action<string>? StatusChanged;
    public event Action? SimulationCompleted;
    public event Action? SimulationStarted;

    public void Start(SleepScenarioType scenario, DateTime? sessionStart = null)
    {
        lock (_lock)
        {
            if (IsRunning) return;
            IsRunning = true;
        }

        _scenario = scenario;
        _sessionStart = sessionStart ?? DateTime.Now;
        _elapsedSeconds = 0;
        _drugEffectLevel = 0;
        _desatCount = 0;

        InitScenario(scenario);

        _cts = new CancellationTokenSource();
        _runTask = Task.Run(() => RunLoop(_cts.Token));
        SimulationStarted?.Invoke();
        StatusChanged?.Invoke($"模拟已启动: {GetScenarioName(scenario)}");
    }

    public void Stop()
    {
        lock (_lock)
        {
            if (!IsRunning) return;
            _cts?.Cancel();
            IsRunning = false;
        }
        StatusChanged?.Invoke("模拟已停止");
        SimulationCompleted?.Invoke();
    }

    public void SetSpeed(double speed)
    {
        PlaybackSpeed = speed;
        StatusChanged?.Invoke($"模拟速度: {speed}x");
    }

    private void InitScenario(SleepScenarioType scenario)
    {
        switch (scenario)
        {
            case SleepScenarioType.NaturalSleep:
                _sleepOnsetMinute = 5 + _rng.Next(0, 8);   // 5-13 min
                break;
            case SleepScenarioType.DISE:
                _sleepOnsetMinute = 1;
                _drugEffectLevel = 0;
                break;
            case SleepScenarioType.OSAScreening:
                _sleepOnsetMinute = 8 + _rng.Next(0, 6);    // 8-14 min
                _desatCount = 0;
                break;
            case SleepScenarioType.ConsciousSedation:
                _sleepOnsetMinute = 2;
                _drugEffectLevel = 0.5;
                break;
        }
    }

    private async Task RunLoop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            double tickMs = 1000.0 / PlaybackSpeed;
            double minute = _elapsedSeconds / 60.0;
            var (stage, _) = DetermineSleepStage(minute);
            var result = GenerateEpoch(_elapsedSeconds, minute, stage);
            ResultGenerated?.Invoke(result);

            if (_elapsedSeconds % 5 == 0)
            {
                var vitals = GenerateVitalSigns(result.Timestamp, stage);
                VitalSignsUpdated?.Invoke(vitals);
            }

            _elapsedSeconds++;
            if (_elapsedSeconds % 30 == 0)
                MaybeGenerateClinicalEvent();

            try { await Task.Delay((int)tickMs, ct); }
            catch (OperationCanceledException) { break; }
        }

        lock (_lock) IsRunning = false;
        SimulationCompleted?.Invoke();
    }

    private ProcessedEEGResult GenerateEpoch(int second, double minute, SleepStage stage)
    {

        double delta, theta, alpha, beta, gamma;
        (delta, theta, alpha, beta, gamma) = GenerateBandPowers(stage);

        double bis = GenerateBIS(stage, minute);
        double? se = GenerateStateEntropy(stage);
        double? re = se.HasValue ? se.Value + (_rng.NextDouble() * stage switch
        {
            SleepStage.Wake => 8.0,
            SleepStage.N1 => 4.0,
            SleepStage.N2 => 2.5,
            SleepStage.N3 => 1.0,
            SleepStage.REM => 2.0,
            _ => 3.0
        }) : null;

        double reSeDiff = re.HasValue && se.HasValue ? re.Value - se.Value : 0;

        double spindleDensity = stage switch
        {
            SleepStage.N2 => 3.0 + _rng.NextDouble() * 5.0,
            SleepStage.N3 => _rng.NextDouble() * 1.5,
            _ => _rng.NextDouble() * 0.5
        };

        bool isLikelySleep = stage is SleepStage.N1 or SleepStage.N2 or SleepStage.N3 or SleepStage.REM;

        double spo2 = GenerateSpO2(stage, second);
        double hr = GenerateHeartRate(stage);

        // Drug effect: gradually deepens sedation, then wears off
        if (_scenario is SleepScenarioType.DISE or SleepScenarioType.ConsciousSedation)
            _drugEffectLevel = ComputeDrugEffect(minute);

        // Synthetic EEG waveforms (128 samples @ 128 Hz = 1 second)
        int nSamples = 128;
        var (raw, filtered, deltaW, thetaW, alphaW, betaW, gammaW) = SynthesizeEEG(stage, nSamples);

        return new ProcessedEEGResult
        {
            Timestamp = _sessionStart.AddSeconds(second),
            BIS = bis,
            SQI = 0.85 + _rng.NextDouble() * 0.14,
            StateEntropy = se,
            ResponseEntropy = re,
            DeltaPower = delta,
            ThetaPower = theta,
            AlphaPower = alpha,
            BetaPower = beta,
            GammaPower = gamma,
            SpindleDensity = spindleDensity,
            IsLikelySleep = isLikelySleep,
            TotalSpindleCount = (int)(spindleDensity * 30),
            SpO2 = spo2,
            HeartRate = hr,
            HRV_RMSSD = stage switch
            {
                SleepStage.N3 => 40 + _rng.NextDouble() * 20,
                SleepStage.REM => 20 + _rng.NextDouble() * 15,
                _ => 25 + _rng.NextDouble() * 15
            },
            EegDominantHz = stage switch
            {
                SleepStage.N3 => 1.0 + _rng.NextDouble() * 2.0,
                SleepStage.N2 => 3.0 + _rng.NextDouble() * 5.0,
                SleepStage.N1 => 5.0 + _rng.NextDouble() * 3.0,
                SleepStage.REM => 3.0 + _rng.NextDouble() * 7.0,
                SleepStage.Wake => 8.0 + _rng.NextDouble() * 12.0,
                _ => 5.0 + _rng.NextDouble() * 10.0
            },
            EegAmplitudeUv = stage switch
            {
                SleepStage.N3 => 40 + _rng.NextDouble() * 60,
                SleepStage.N2 => 15 + _rng.NextDouble() * 25,
                _ => 5 + _rng.NextDouble() * 15
            },
            EegTonalRatio = _rng.NextDouble() * 0.05,
            HwClippingPct = 0,
            HwIsSaturated = false,
            HwDcOffsetUv = _rng.NextDouble() * 2.0 - 1.0,
            HwAdcRangeUv = 250,
            FNox = bis < 60 ? 20 + _rng.NextDouble() * 40 : 50 + _rng.NextDouble() * 30,
            RawEEG = raw,
            FilteredEEG = filtered,
            DeltaWave = deltaW,
            ThetaWave = thetaW,
            AlphaWave = alphaW,
            BetaWave = betaW,
            GammaWave = gammaW,
        };
    }

    private (double[] raw, double[] filtered, double[] delta, double[] theta, double[] alpha, double[] beta, double[] gamma)
        SynthesizeEEG(SleepStage stage, int n)
    {
        double fs = 128.0;
        var raw = new double[n];
        var filtered = new double[n];
        var deltaW = new double[n];
        var thetaW = new double[n];
        var alphaW = new double[n];
        var betaW = new double[n];
        var gammaW = new double[n];

        // Amplitude weights per stage (uV)
        double dAmp, tAmp, aAmp, bAmp, gAmp, noiseAmp;
        switch (stage)
        {
            case SleepStage.Wake:
                dAmp = 5; tAmp = 5; aAmp = 12; bAmp = 18; gAmp = 6; noiseAmp = 4; break;
            case SleepStage.N1:
                dAmp = 10; tAmp = 15; aAmp = 8; bAmp = 6; gAmp = 3; noiseAmp = 2; break;
            case SleepStage.N2:
                dAmp = 20; tAmp = 10; aAmp = 8; bAmp = 5; gAmp = 2; noiseAmp = 2; break;
            case SleepStage.N3:
                dAmp = 50; tAmp = 8; aAmp = 4; bAmp = 3; gAmp = 1; noiseAmp = 2; break;
            case SleepStage.REM:
                dAmp = 8; tAmp = 12; aAmp = 10; bAmp = 8; gAmp = 3; noiseAmp = 3; break;
            default:
                dAmp = 10; tAmp = 10; aAmp = 8; bAmp = 8; gAmp = 3; noiseAmp = 2; break;
        }

        double phaseD = _rng.NextDouble() * Math.PI * 2;
        double phaseT = _rng.NextDouble() * Math.PI * 2;
        double phaseA = _rng.NextDouble() * Math.PI * 2;
        double phaseB = _rng.NextDouble() * Math.PI * 2;
        double phaseG = _rng.NextDouble() * Math.PI * 2;

        for (int i = 0; i < n; i++)
        {
            double t = i / fs;
            double d = dAmp * Math.Sin(2 * Math.PI * 2.0 * t + phaseD);   // 2 Hz delta
            double th = tAmp * Math.Sin(2 * Math.PI * 6.0 * t + phaseT);   // 6 Hz theta
            double a = aAmp * Math.Sin(2 * Math.PI * 10.0 * t + phaseA);   // 10 Hz alpha
            double b = bAmp * Math.Sin(2 * Math.PI * 20.0 * t + phaseB);   // 20 Hz beta
            double g = gAmp * Math.Sin(2 * Math.PI * 38.0 * t + phaseG);   // 38 Hz gamma
            double noise = noiseAmp * (_rng.NextDouble() * 2 - 1);

            // Spindle burst in N2: 12-15 Hz waxing/waning
            double spindle = 0;
            if (stage == SleepStage.N2 && _rng.NextDouble() < 0.08)
            {
                double env = Math.Sin(i * Math.PI / 40) * (i < 40 ? 1 : 0); // 0.3s envelope
                spindle = env * 15 * Math.Sin(2 * Math.PI * 13.5 * t);
            }

            deltaW[i] = d;
            thetaW[i] = th;
            alphaW[i] = a;
            betaW[i] = b;
            gammaW[i] = g;
            filtered[i] = d + th + a + b + g + spindle;
            raw[i] = filtered[i] + noise;
        }

        return (raw, filtered, deltaW, thetaW, alphaW, betaW, gammaW);
    }

    private VitalSignsSnapshot GenerateVitalSigns(DateTime timestamp, SleepStage stage)
    {
        int hr = (int)GenerateHeartRate(stage);
        int spo2 = (int)Math.Round(GenerateSpO2(stage, _elapsedSeconds));

        double minute = _elapsedSeconds / 60.0;
        double sbp, dbp;

        switch (stage)
        {
            case SleepStage.N3:
                sbp = 95 + _rng.NextDouble() * 12; dbp = 55 + _rng.NextDouble() * 8; break;
            case SleepStage.N2:
                sbp = 100 + _rng.NextDouble() * 12; dbp = 60 + _rng.NextDouble() * 8; break;
            case SleepStage.REM:
                sbp = 105 + _rng.NextDouble() * 18; dbp = 62 + _rng.NextDouble() * 12; break;
            case SleepStage.Wake:
                sbp = 115 + _rng.NextDouble() * 15; dbp = 70 + _rng.NextDouble() * 10; break;
            default:
                sbp = 108 + _rng.NextDouble() * 12; dbp = 65 + _rng.NextDouble() * 8; break;
        }

        // Drug effect: mild BP reduction
        sbp -= _drugEffectLevel * 10;
        dbp -= _drugEffectLevel * 5;

        int rr = stage switch
        {
            SleepStage.N3 => 10 + _rng.Next(0, 3),
            SleepStage.N2 => 12 + _rng.Next(0, 3),
            SleepStage.REM => 13 + _rng.Next(0, 6),
            SleepStage.Wake => 14 + _rng.Next(0, 4),
            _ => 12 + _rng.Next(0, 4)
        };

        int etco2 = 35 + _rng.Next(0, 6);
        // OSA: EtCO2 rises during events
        if (_scenario == SleepScenarioType.OSAScreening && _desatCount > 0)
            etco2 += 3 + _rng.Next(0, 5);

        // DISE: respiratory depression
        if (_scenario == SleepScenarioType.DISE && _drugEffectLevel > 0.5)
            etco2 += (int)(_drugEffectLevel * 10);

        double temp = 36.5 + _rng.NextDouble() * 0.8;
        // N3: slight temp drop
        if (stage == SleepStage.N3) temp -= 0.3;

        return new VitalSignsSnapshot
        {
            Timestamp = timestamp,
            HeartRate = hr,
            SpO2 = spo2,
            SystolicBP = (int)Math.Round(sbp),
            DiastolicBP = (int)Math.Round(dbp),
            RespiratoryRate = rr,
            EtCO2 = etco2,
            Temperature = Math.Round(temp, 1)
        };
    }

    private (SleepStage stage, double confidence) DetermineSleepStage(double minute)
    {
        int sleepMinute = _sleepOnsetMinute;
        double cycleLength = 85 + _rng.NextDouble() * 15; // 85-100 min cycles

        if (minute < sleepMinute)
        {
            // Pre-sleep wakefulness
            if (_scenario is SleepScenarioType.DISE or SleepScenarioType.ConsciousSedation)
                return _drugEffectLevel > 0.4 ? (SleepStage.N1, 0.7) : (SleepStage.Wake, 0.9);
            return (SleepStage.Wake, 0.95);
        }

        double sleepElapsed = minute - sleepMinute;
        int cycleIndex = (int)(sleepElapsed / cycleLength);
        double cyclePos = (sleepElapsed % cycleLength) / cycleLength; // 0.0 - 1.0

        // Each cycle: Wake/arousal → N1 → N2 → N3 → N2 → REM → (brief arousal)
        // Early cycles (0-2): more N3
        // Later cycles (3+): less N3, more REM

        double n3Weight = Math.Max(0, 1.0 - cycleIndex * 0.35);
        double remWeight = Math.Min(1.0, cycleIndex * 0.3);

        if (_scenario is SleepScenarioType.DISE or SleepScenarioType.ConsciousSedation)
        {
            // Drug-induced: predominantly N2/N3, suppressed REM
            n3Weight = _drugEffectLevel * 1.2;
            remWeight = 0;
        }

        if (cyclePos < 0.02) return (SleepStage.Wake, 0.85);
        if (cyclePos < 0.06) return (SleepStage.N1, 0.7);
        if (cyclePos < 0.45) return _rng.NextDouble() < n3Weight * 0.3
            ? (SleepStage.N3, 0.75) : (SleepStage.N2, 0.8);
        if (cyclePos < 0.55) return _rng.NextDouble() < n3Weight * 0.6
            ? (SleepStage.N3, 0.8) : (SleepStage.N2, 0.75);
        if (cyclePos < 0.75) return (SleepStage.N2, 0.7);
        if (cyclePos < 0.92) return _rng.NextDouble() < remWeight
            ? (SleepStage.REM, 0.7) : (SleepStage.N2, 0.65);
        // Brief arousal at end of cycle
        return _rng.NextDouble() < 0.6 ? (SleepStage.Wake, 0.6) : (SleepStage.N1, 0.55);
    }

    private (double delta, double theta, double alpha, double beta, double gamma) GenerateBandPowers(SleepStage stage)
    {
        double d, t, a, b, g;
        switch (stage)
        {
            case SleepStage.Wake:
                d = 0.10 + _rng.NextDouble() * 0.10;
                t = 0.08 + _rng.NextDouble() * 0.08;
                a = 0.15 + _rng.NextDouble() * 0.15;
                b = 0.20 + _rng.NextDouble() * 0.25;
                g = 0.10 + _rng.NextDouble() * 0.15;
                break;
            case SleepStage.N1:
                d = 0.15 + _rng.NextDouble() * 0.10;
                t = 0.20 + _rng.NextDouble() * 0.15;
                a = 0.10 + _rng.NextDouble() * 0.08;
                b = 0.08 + _rng.NextDouble() * 0.07;
                g = 0.04 + _rng.NextDouble() * 0.05;
                break;
            case SleepStage.N2:
                d = 0.25 + _rng.NextDouble() * 0.10;
                t = 0.12 + _rng.NextDouble() * 0.08;
                a = 0.10 + _rng.NextDouble() * 0.06;
                b = 0.06 + _rng.NextDouble() * 0.05;
                g = 0.02 + _rng.NextDouble() * 0.04;
                break;
            case SleepStage.N3:
                d = 0.45 + _rng.NextDouble() * 0.20;
                t = 0.06 + _rng.NextDouble() * 0.06;
                a = 0.04 + _rng.NextDouble() * 0.04;
                b = 0.02 + _rng.NextDouble() * 0.03;
                g = 0.01 + _rng.NextDouble() * 0.02;
                break;
            case SleepStage.REM:
                d = 0.10 + _rng.NextDouble() * 0.08;
                t = 0.15 + _rng.NextDouble() * 0.12;
                a = 0.15 + _rng.NextDouble() * 0.10;
                b = 0.08 + _rng.NextDouble() * 0.08;
                g = 0.03 + _rng.NextDouble() * 0.05;
                break;
            default:
                d = 0.20; t = 0.15; a = 0.10; b = 0.10; g = 0.05;
                break;
        }

        // Normalize to sum ~1.0
        double sum = d + t + a + b + g;
        return (d / sum, t / sum, a / sum, b / sum, g / sum);
    }

    private double GenerateBIS(SleepStage stage, double minute)
    {
        double baseBis = stage switch
        {
            SleepStage.Wake => 93 + _rng.NextDouble() * 5,
            SleepStage.N1 => 78 + _rng.NextDouble() * 12,
            SleepStage.N2 => 55 + _rng.NextDouble() * 18,
            SleepStage.N3 => 35 + _rng.NextDouble() * 18,
            SleepStage.REM => 65 + _rng.NextDouble() * 18,
            _ => 85
        };

        // Drug effect lowers BIS further
        baseBis -= _drugEffectLevel * 25;

        // Slow variation (sinusoidal drift over minutes)
        baseBis += Math.Sin(minute * 0.03) * 3.0;

        return Math.Clamp(baseBis, 15, 98);
    }

    private double? GenerateStateEntropy(SleepStage stage)
    {
        double baseSE = stage switch
        {
            SleepStage.Wake => 75 + _rng.NextDouble() * 15,
            SleepStage.N1 => 55 + _rng.NextDouble() * 20,
            SleepStage.N2 => 35 + _rng.NextDouble() * 20,
            SleepStage.N3 => 10 + _rng.NextDouble() * 15,
            SleepStage.REM => 30 + _rng.NextDouble() * 25,
            _ => 50
        };
        return Math.Clamp(baseSE, 0, 91);
    }

    private double GenerateSpO2(SleepStage stage, int second)
    {
        double baseline = 97.0;
        double spo2 = baseline + (_rng.NextDouble() - 0.5) * 3.0; // 95.5-98.5%

        // OSA desaturation events: periodic drops during N2/N3/REM
        if (_scenario == SleepScenarioType.OSAScreening && stage is SleepStage.N2 or SleepStage.N3 or SleepStage.REM)
        {
            double minute = second / 60.0;
            // Event every 15-40 minutes, lasting 20-60 seconds
            double eventInterval = 15 + (_desatCount * 7) % 25;
            int eventStartSecond = (int)((_desatCount + 1) * eventInterval * 60);
            int eventDuration = 20 + _rng.Next(0, 40);

            if (second >= eventStartSecond && second < eventStartSecond + eventDuration)
            {
                double progress = (second - eventStartSecond) / (double)eventDuration;
                double drop = Math.Sin(progress * Math.PI) * (8 + _rng.NextDouble() * 12); // 8-20% drop
                spo2 = baseline - drop;
                if (progress > 0.95) _desatCount++;
            }
        }

        // DISE: mild desaturation during deep sedation
        if (_scenario == SleepScenarioType.DISE && _drugEffectLevel > 0.6)
            spo2 -= _drugEffectLevel * 4.0;

        return Math.Clamp(spo2, 70, 100);
    }

    private double GenerateHeartRate(SleepStage stage)
    {
        double baseHR = stage switch
        {
            SleepStage.N3 => 50 + _rng.NextDouble() * 10,
            SleepStage.N2 => 55 + _rng.NextDouble() * 10,
            SleepStage.REM => 58 + _rng.NextDouble() * 15,
            SleepStage.Wake => 62 + _rng.NextDouble() * 18,
            _ => 60 + _rng.NextDouble() * 15
        };
        return Math.Round(baseHR, 1);
    }

    private double ComputeDrugEffect(double minute)
    {
        // Simulate propofol/dexmedetomidine infusion profile
        if (_scenario == SleepScenarioType.DISE)
        {
            // Titration: rapid onset, plateau, then gradual offset
            double onset = Math.Min(1.0, minute / 4.0);            // 0-4 min: rapid rise
            double plateau = minute > 4 && minute < 30 ? 0.9 : 0;  // 4-30 min: maintained
            double offset = minute > 30 ? Math.Max(0, 1.0 - (minute - 30) / 20.0) : 0; // 30-50 min: washout
            return Math.Clamp(onset * 0.9 + plateau * 0.05 + offset, 0, 1);
        }
        if (_scenario == SleepScenarioType.ConsciousSedation)
        {
            // Moderate sedation: bolus → gradual decline
            double onset = Math.Min(1.0, minute / 2.0);
            double decay = Math.Max(0, 1.0 - minute / 60.0);
            return Math.Clamp(onset * 0.5 * decay, 0, 1);
        }
        return 0;
    }

    private void MaybeGenerateClinicalEvent()
    {
        int minute = _elapsedSeconds / 60;
        DateTime now = _sessionStart.AddSeconds(_elapsedSeconds);

        // Sleep onset detection
        if (minute == _sleepOnsetMinute)
        {
            ClinicalEventGenerated?.Invoke(new ClinicalEvent
            {
                Timestamp = now,
                EventType = ClinicalEventType.Custom,
                Label = "睡眠开始",
                IsAutoGenerated = true
            });
        }

        // DISE: drug administration events
        if (_scenario == SleepScenarioType.DISE)
        {
            if (minute == 0)
                EmitEvent(now, "丙泊酚 1.5mg/kg IV 推注开始");
            if (minute == 2)
                EmitEvent(now, "丙泊酚 50mcg/kg/min 维持输注");
            if (minute == 30)
                EmitEvent(now, "输注停止，开始苏醒观察");
        }

        if (_scenario == SleepScenarioType.ConsciousSedation)
        {
            if (minute == 0)
                EmitEvent(now, "咪达唑仑 2mg IV");
            if (minute == 2)
                EmitEvent(now, "芬太尼 50mcg IV");
        }

        // OSA: desaturation alerts
        if (_scenario == SleepScenarioType.OSAScreening && _desatCount > 0 && _desatCount % 3 == 0)
        {
            EmitEvent(now, $"第{_desatCount}次脱氧事件");
        }
    }

    private void EmitEvent(DateTime time, string label)
    {
        ClinicalEventGenerated?.Invoke(new ClinicalEvent
        {
            Timestamp = time,
            EventType = ClinicalEventType.Custom,
            Label = label,
            IsAutoGenerated = true
        });
    }

    public static string GetScenarioName(SleepScenarioType s) => s switch
    {
        SleepScenarioType.NaturalSleep => "自然睡眠（正常成人夜间睡眠）",
        SleepScenarioType.DISE => "药物诱导睡眠内镜 (DISE)",
        SleepScenarioType.OSAScreening => "OSA 筛查（睡眠呼吸暂停）",
        SleepScenarioType.ConsciousSedation => "清醒镇静（咪达唑仑+芬太尼）",
        _ => s.ToString()
    };

    public static string GetScenarioDescription(SleepScenarioType s) => s switch
    {
        SleepScenarioType.NaturalSleep => "模拟正常成人整夜睡眠结构，含4-5个NREM-REM周期、自然入睡和晨间觉醒。",
        SleepScenarioType.DISE => "模拟丙泊酚靶控输注下的药物诱导睡眠，用于上气道塌陷评估。药物起效→维持→苏醒全流程。",
        SleepScenarioType.OSAScreening => "模拟阻塞性睡眠呼吸暂停患者夜间SpO₂反复下降模式，评估氧减指数(ODI)及最低血氧。",
        SleepScenarioType.ConsciousSedation => "模拟咪达唑仑联合芬太尼清醒镇静过程，适用于门诊内镜等操作的镇静深度监测。",
        _ => ""
    };
}
