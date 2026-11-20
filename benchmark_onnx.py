"""CPU benchmark of an ONNX model with ONNX Runtime.
Usage: python benchmark_onnx.py model.onnx [--runs 200] [--warmup 30] [--threads 1] [--batch 1] [--res 32]
Prints one JSON line: latency (mean/median/p95/std, ms), throughput, peak memory (RSS growth + absolute)."""
import argparse, json, os, time
import numpy as np

def _status_mb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key): return int(line.split()[1]) / 1024.0      # kB -> MB
    return 0.0

def rss_mb(): return _status_mb("VmRSS:")
def hwm_mb(): return _status_mb("VmHWM:")   # peak RSS of THIS process (reset on exec, unlike ru_maxrss which a fork inherits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--runs", type=int, default=200); ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--batch", type=int, default=1); ap.add_argument("--res", type=int, default=32)
    a = ap.parse_args()
    import onnxruntime as ort
    base = rss_mb()
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(a.batch, 3, a.res, a.res).astype(np.float32)
    for _ in range(a.warmup): sess.run(None, {name: x})
    ts = []
    for _ in range(a.runs):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts); peak = hwm_mb()
    print(json.dumps(dict(lat_mean_ms=float(ts.mean()), lat_median_ms=float(np.median(ts)), lat_p95_ms=float(np.percentile(ts, 95)),
                          lat_std_ms=float(ts.std()), throughput_ips=float(a.batch * 1000.0 / ts.mean()),
                          peak_mem_mb=float(max(peak - base, 0.0)), peak_rss_mb=float(peak),
                          threads=a.threads, batch=a.batch, runs=a.runs)))
main()
