"""test_python_service.py — end-to-end health check of the EEG inference service.

Feeds SYNTHETIC single-channel EEG (no hardware needed) to /process and verifies every
output the WPF relies on: filtered waveform, 5 band waves, band powers, DSA, SQI, BIS.
Use this to prove the Python service works independently of the ADS1299 board.

Usage: python scripts/test_python_service.py [base_url]
"""
import sys, time, json
import numpy as np
import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/").rstrip("/")
FS = 250
N = FS               # 1-second chunk
CHUNKS = 12          # ~12 s so the BIS buffer/calibration warms up

def banner(t): print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)

def make_chunk(t0):
    """Realistic-ish awake EEG: alpha+beta+noise + 50 Hz mains, single channel."""
    t = (t0 + np.arange(N)) / FS
    eeg = (12*np.sin(2*np.pi*10*t)      # alpha
           + 6*np.sin(2*np.pi*20*t)     # beta
           + 4*np.sin(2*np.pi*2*t)      # delta
           + 5*np.random.randn(N))      # broadband
    mains = 120*np.sin(2*np.pi*50*t)    # heavy 50 Hz, like the real board
    sig = eeg + mains
    return sig.reshape(-1, 1)           # (n_samples, 1)

# ── 1. health ──────────────────────────────────────────────────────────────
banner("1) /health")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    print(f"  HTTP {r.status_code}  {r.text}")
    health_ok = r.status_code == 200 and r.json().get("model_loaded") is True
    print(f"  -> service up: {r.status_code==200}   model loaded: {r.json().get('model_loaded')}")
except Exception as e:
    print(f"  FAILED to reach service: {e}")
    print("\n服务未运行。先启动: cd EEGMonitor/EEGProcessingService && python main.py")
    sys.exit(1)

# reset session state
try: requests.post(f"{BASE}/reset", timeout=5); print("  session reset OK")
except Exception as e: print(f"  reset failed: {e}")

# ── 2. feed chunks ───────────────────────────────────────────────────────────
banner("2) POST /process  (synthetic EEG, 250Hz single channel)")
last = None
bis_seen = False
t0 = 0
for i in range(CHUNKS):
    chunk = make_chunk(t0); t0 += N
    payload = {"sample_rate": FS, "channel_count": 1, "eeg_data": chunk.tolist()}
    try:
        r = requests.post(f"{BASE}/process", json=payload, timeout=15)
    except Exception as e:
        print(f"  chunk {i}: request FAILED: {e}"); continue
    if r.status_code != 200:
        print(f"  chunk {i}: HTTP {r.status_code}  {r.text[:200]}"); continue
    j = r.json(); last = j
    bis = j.get("bis")
    if bis is not None: bis_seen = True
    print(f"  chunk {i:2d}: bis={bis if bis is None else round(bis,1)!s:>5}  sqi={round(j.get('sqi',0)):>3}  "
          f"filtered={len(j.get('filtered_eeg',[])):>3}  "
          f"δ={j.get('delta_power',0):.2f} θ={j.get('theta_power',0):.2f} "
          f"α={j.get('alpha_power',0):.2f} β={j.get('beta_power',0):.2f} γ={j.get('gamma_power',0):.2f}  "
          f"domHz={j.get('eeg_dominant_hz',0):.1f}")
    time.sleep(0.05)

# ── 3. verdict ───────────────────────────────────────────────────────────────
banner("3) VERDICT")
if last is None:
    print("  [FAIL] 服务无有效响应"); sys.exit(1)
checks = {
    "health 200 + model loaded": health_ok,
    "filtered_eeg 非空 (波形)": len(last.get("filtered_eeg", [])) > 0,
    "5 个成分波非空": all(len(last.get(b, [])) > 0 for b in
                         ["delta_wave","theta_wave","alpha_wave","beta_wave","gamma_wave"]),
    "频带功率和≈1": abs(sum(last.get(b,0) for b in
                         ["delta_power","theta_power","alpha_power","beta_power","gamma_power"]) - 1.0) < 0.05,
    "DSA 矩阵非空": len(last.get("dsa_matrix", [])) > 0,
    "SQI 在 0-100": 0 <= last.get("sqi", -1) <= 100,
    "BIS 已输出 (非NaN)": bis_seen,
    "工频已抑制 (主频<45Hz)": last.get("eeg_dominant_hz", 99) < 45,
}
for name, ok in checks.items():
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {name}")
allok = all(checks.values())
print("\n" + ("[PASS] Python 服务完全正常 —— 问题不在服务，在于没有串口数据输入。"
              if allok else "[FAIL] Python 服务存在问题，见上面失败项。"))
