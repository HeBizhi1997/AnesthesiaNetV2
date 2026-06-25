namespace NSMMonitor.Services;

/// <summary>
/// EEG 频带成分分离：对一段脑电样本做 FFT 带通（砖墙滤波）后逆变换，
/// 得到指定频带（δ/θ/α/β）的时域分量波形，用于成分分离显示。
/// </summary>
public static class EegBandSeparator
{
    /// <summary>需要分离的频带（不含 γ）。</summary>
    public static readonly (string Name, double Lo, double Hi)[] Bands =
    {
        ("δ Delta", 1, 3),
        ("θ Theta", 4, 7),
        ("α Alpha", 8, 13),
        ("β Beta", 14, 30),
    };

    /// <summary>对 signal 做 [lo, hi] Hz 带通，返回与输入等长的时域分量。</summary>
    public static double[] BandPass(double[] signal, double fs, double lo, double hi)
    {
        int n = signal.Length;
        if (n < 2 || fs <= 0) return (double[])signal.Clone();

        int m = 1;
        while (m < n) m <<= 1;        // 补零到 2 的幂

        var re = new double[m];
        var im = new double[m];

        // 去均值，避免直流分量主导
        double mean = 0;
        for (int i = 0; i < n; i++) mean += signal[i];
        mean /= n;
        for (int i = 0; i < n; i++) re[i] = signal[i] - mean;

        Fft(re, im, inverse: false);

        // 频域砖墙带通：bin k 对应频率 |f| = (k 或 m-k)·fs/m
        for (int k = 0; k < m; k++)
        {
            double f = (k <= m / 2 ? k : m - k) * fs / m;
            if (f < lo || f > hi) { re[k] = 0; im[k] = 0; }
        }

        Fft(re, im, inverse: true);

        var outp = new double[n];
        Array.Copy(re, outp, n);
        return outp;
    }

    /// <summary>迭代式 Cooley-Tukey FFT（原地，长度须为 2 的幂）。</summary>
    private static void Fft(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;

        // 位反转置换
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * Math.PI / len * (inverse ? 1 : -1);
            double wLenRe = Math.Cos(ang), wLenIm = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double wRe = 1, wIm = 0;
                for (int k = 0; k < len / 2; k++)
                {
                    int a = i + k, b = i + k + len / 2;
                    double vRe = re[b] * wRe - im[b] * wIm;
                    double vIm = re[b] * wIm + im[b] * wRe;
                    re[b] = re[a] - vRe; im[b] = im[a] - vIm;
                    re[a] += vRe;        im[a] += vIm;
                    double nwRe = wRe * wLenRe - wIm * wLenIm;
                    wIm = wRe * wLenIm + wIm * wLenRe;
                    wRe = nwRe;
                }
            }
        }

        if (inverse)
            for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}
