namespace EEGRecorder.Dsp;

/// <summary>Direct-Form-II-Transposed biquad (RBJ cookbook). Causal — used for the QC display only;
/// recorded data stays raw.</summary>
public sealed class Biquad
{
    private double _b0, _b1, _b2, _a1, _a2, _z1, _z2;

    public double Process(double x)
    {
        double y = _b0 * x + _z1;
        _z1 = _b1 * x - _a1 * y + _z2;
        _z2 = _b2 * x - _a2 * y;
        return y;
    }

    public void Reset() => _z1 = _z2 = 0;

    private void Normalize(double b0, double b1, double b2, double a0, double a1, double a2)
    {
        _b0 = b0 / a0; _b1 = b1 / a0; _b2 = b2 / a0; _a1 = a1 / a0; _a2 = a2 / a0;
    }

    public static Biquad HighPass(double f0, double fs, double q = 0.707)
    {
        double w0 = 2 * Math.PI * f0 / fs, c = Math.Cos(w0), s = Math.Sin(w0), alpha = s / (2 * q);
        var bq = new Biquad();
        bq.Normalize((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + alpha, -2 * c, 1 - alpha);
        return bq;
    }

    public static Biquad LowPass(double f0, double fs, double q = 0.707)
    {
        double w0 = 2 * Math.PI * f0 / fs, c = Math.Cos(w0), s = Math.Sin(w0), alpha = s / (2 * q);
        var bq = new Biquad();
        bq.Normalize((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + alpha, -2 * c, 1 - alpha);
        return bq;
    }

    public static Biquad Notch(double f0, double fs, double q = 30)
    {
        double w0 = 2 * Math.PI * f0 / fs, c = Math.Cos(w0), s = Math.Sin(w0), alpha = s / (2 * q);
        var bq = new Biquad();
        bq.Normalize(1, -2 * c, 1, 1 + alpha, -2 * c, 1 - alpha);
        return bq;
    }
}

/// <summary>EEG display chain: 0.5 Hz high-pass (DC/drift) → 50 Hz notch (mains) → 45 Hz low-pass.
/// Cosmetic only, so the operator can see real EEG vs lead-off noise while recording RAW.</summary>
public sealed class EegDisplayFilter
{
    private readonly Biquad _hp, _notch, _lp;

    public EegDisplayFilter(double fs)
    {
        _hp = Biquad.HighPass(0.5, fs);
        _notch = Biquad.Notch(50, fs, 30);
        _lp = Biquad.LowPass(45, fs);
    }

    public double Process(double x) => _lp.Process(_notch.Process(_hp.Process(x)));
    public void Reset() { _hp.Reset(); _notch.Reset(); _lp.Reset(); }
}

/// <summary>PPG display chain: 0.5–5 Hz band-pass to pull the pulse wave off the big DC pedestal.</summary>
public sealed class PpgDisplayFilter
{
    private readonly Biquad _hp, _lp;

    public PpgDisplayFilter(double fs)
    {
        _hp = Biquad.HighPass(0.5, fs);
        _lp = Biquad.LowPass(5.0, fs);
    }

    public double Process(double x) => _lp.Process(_hp.Process(x));
    public void Reset() { _hp.Reset(); _lp.Reset(); }
}
