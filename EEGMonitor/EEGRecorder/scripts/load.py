"""load.py — read an EEGRecorder session folder (eeg.bin + ppg.bin + meta.json).

Usage:
    python load.py <session_folder> [--plot]

Returns (and prints a summary of):
    eeg : float32 array, CH0 microvolts, uniform sample rate = meta.eeg.measured_rate_hz
    ppg : structured array [ticks(i8), ir(i4), red(i4), spo2(u1), hr(u1)]  (ticks = .NET DateTime ticks)
"""
import sys, os, json
import numpy as np

PPG_DTYPE = np.dtype([("ticks", "<i8"), ("ir", "<i4"), ("red", "<i4"),
                      ("spo2", "u1"), ("hr", "u1")])          # packed, 18 bytes/record


def load(folder):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    eeg = np.fromfile(os.path.join(folder, "eeg.bin"), dtype="<f4")
    ppg_path = os.path.join(folder, "ppg.bin")
    ppg = np.fromfile(ppg_path, dtype=PPG_DTYPE) if os.path.getsize(ppg_path) else np.empty(0, PPG_DTYPE)
    return meta, eeg, ppg


def dotnet_ticks_to_unix(ticks):
    # .NET ticks: 100ns since 0001-01-01. Convert to unix seconds.
    return (ticks - 621355968000000000) / 1e7


def main():
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit
    folder = sys.argv[1]
    meta, eeg, ppg = load(folder)

    fs = meta["eeg"].get("measured_rate_hz") or meta["eeg"]["nominal_rate_hz"]
    print(f"== {os.path.basename(folder)} ==")
    print(f"subject : {meta.get('subject')}   note: {meta.get('note')}")
    print(f"started : {meta.get('started')}")
    print(f"EEG     : {eeg.size} samples @ ~{fs:.1f} Hz  = {eeg.size / fs / 60:.1f} min"
          f"   (gain {meta['eeg'].get('gain')}, {'diff' if meta['eeg'].get('differential') else 'single'})")
    if eeg.size:
        print(f"          uV range {eeg.min():.1f} .. {eeg.max():.1f},  std {eeg.std():.1f}")
    if ppg.size:
        hr = ppg["hr"][ppg["hr"] > 0]
        sp = ppg["spo2"][ppg["spo2"] > 0]
        print(f"PPG     : {ppg.size} frames @ ~{meta['ppg'].get('measured_rate_hz')} Hz"
              f"   HR~{np.median(hr) if hr.size else 0:.0f}  SpO2~{np.median(sp) if sp.size else 0:.0f}%")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt
        n = min(eeg.size, int(fs * 10))
        fig, ax = plt.subplots(2, 1, figsize=(11, 6))
        ax[0].plot(np.arange(n) / fs, eeg[:n], lw=0.6, color="#38bdf8")
        ax[0].set_title("EEG (first 10 s, raw uV)"); ax[0].set_xlabel("s")
        if ppg.size:
            m = min(ppg.size, int(125 * 10))
            ax[1].plot(ppg["ir"][:m], lw=0.6, color="#f87171")
            ax[1].set_title("PPG IR (first ~10 s, raw)")
        fig.tight_layout(); plt.show()

    return meta, eeg, ppg


if __name__ == "__main__":
    main()
