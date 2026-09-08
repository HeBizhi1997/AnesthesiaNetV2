using NSMMonitor.Models;

namespace NSMMonitor.Services.Ppg;

/// <summary>
/// AFE4490 17 字节帧的流式解析器，严格实现 serial-protocol.md §6.1 / §6.2。
///
/// 单独成类是为了可测：帧同步是整条链路里最容易出错的一环 —— 负载为任意二进制，
/// 同步字 0A FA 在 IR/RED 的 int32 里偶现的概率不可忽略，协议因此明确
/// <b>禁止只匹配帧头不验帧尾</b>。这里把逻辑与 SerialPort 解耦，便于用构造好的字节流验证：
/// 启动横幅、逐字节喂入、假同步字、半帧截断等场景。
/// </summary>
public sealed class PpgFrameParser
{
    public const int FrameSize = 17;
    private const byte SYNC0 = 0x0A, SYNC1 = 0xFA;
    private const byte LEN_L = 0x0A, LEN_H = 0x00, TYPE = 0x02;
    private const byte STOP0 = 0x00, STOP1 = 0x0B;

    /// <summary>缓冲上限：始终同步不上时（波特率错等）避免无限增长。</summary>
    private const int MaxBuffer = 8192;

    private readonly List<byte> _buffer = new(4096);

    /// <summary>
    /// 严格模式下额外校验偏移 15（协议规定的 STOP0 = 0x00）。
    /// 老固件把该字节当状态位用，此时严格校验会把每一帧都判坏，需要关掉。
    /// </summary>
    public bool Strict { get; set; } = true;

    public long FramesDecoded { get; private set; }
    public long BadFrames { get; private set; }

    public void Reset()
    {
        _buffer.Clear();
        FramesDecoded = 0;
        BadFrames = 0;
    }

    /// <summary>喂入一段刚读到的字节，返回本次凑齐的完整帧（可能为空）。</summary>
    public IReadOnlyList<PpgFrame> Feed(ReadOnlySpan<byte> data)
    {
        foreach (var b in data) _buffer.Add(b);

        var frames = new List<PpgFrame>();
        while (true)
        {
            int i = IndexOfSync();
            if (i < 0)
            {
                // 未找到同步字：保留末尾 1 字节 —— 同步字可能正好跨在两次读取的边界上
                if (_buffer.Count > 1) _buffer.RemoveRange(0, _buffer.Count - 1);
                break;
            }
            if (_buffer.Count - i < FrameSize)
            {
                if (i > 0) _buffer.RemoveRange(0, i);   // 帧还没收全，先把同步字之前的垃圾丢掉
                break;
            }

            if (IsValidFrame(i))
            {
                frames.Add(new PpgFrame(
                    DateTime.Now,
                    Int32LE(i + 5),
                    Int32LE(i + 9),
                    _buffer[i + 13],
                    _buffer[i + 14]));
                _buffer.RemoveRange(0, i + FrameSize);
                FramesDecoded++;
            }
            else
            {
                // 假同步字：只丢 2 字节，从下一字节继续搜索（§6.1 第 4 条）。
                // 丢整帧会把紧随其后的真帧一起吃掉。
                _buffer.RemoveRange(0, i + 2);
                BadFrames++;
            }
        }

        if (_buffer.Count > MaxBuffer) _buffer.Clear();
        return frames;
    }

    private int IndexOfSync()
    {
        for (int i = 0; i + 1 < _buffer.Count; i++)
            if (_buffer[i] == SYNC0 && _buffer[i + 1] == SYNC1) return i;
        return -1;
    }

    private bool IsValidFrame(int i) =>
        _buffer[i + 2] == LEN_L &&
        _buffer[i + 3] == LEN_H &&
        _buffer[i + 4] == TYPE &&
        (!Strict || _buffer[i + 15] == STOP0) &&
        _buffer[i + 16] == STOP1;

    private int Int32LE(int at) =>
        _buffer[at] | (_buffer[at + 1] << 8) | (_buffer[at + 2] << 16) | (_buffer[at + 3] << 24);

    /// <summary>按协议 §4 组帧，供模拟器与测试构造字节流。</summary>
    public static byte[] BuildFrame(int ir, int red, byte spo2 = 0, byte hr = 0)
    {
        var f = new byte[FrameSize];
        f[0] = SYNC0; f[1] = SYNC1; f[2] = LEN_L; f[3] = LEN_H; f[4] = TYPE;
        WriteInt32LE(f, 5, ir);
        WriteInt32LE(f, 9, red);
        f[13] = spo2; f[14] = hr; f[15] = STOP0; f[16] = STOP1;
        return f;
    }

    private static void WriteInt32LE(byte[] b, int at, int v)
    {
        b[at] = (byte)v;
        b[at + 1] = (byte)(v >> 8);
        b[at + 2] = (byte)(v >> 16);
        b[at + 3] = (byte)(v >> 24);
    }
}
