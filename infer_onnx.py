"""Stand-alone CPU inference with ONNX Runtime for ALL exported models (6 students + NAS model).
Only needs: numpy, pillow, onnxruntime.

  python infer_onnx.py --image cat.png                       # run every model in ./onnx on one image
  python infer_onnx.py --image cat.png --model onnx/kd_mobilenetv2.onnx
  python infer_onnx.py --test-cifar                          # accuracy of every model on its CIFAR test set (needs torchvision)
"""
import argparse, json, glob, os
import numpy as np, onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = {"cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)), "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}

def dataset_of(path): return "cifar10" if os.path.basename(path).startswith("nas") else "cifar100"

def normalize(batch_uint8, ds):
    m, s = STATS[ds]; x = batch_uint8.astype(np.float32) / 255.0
    return np.ascontiguousarray(((x - np.array(m, np.float32)) / np.array(s, np.float32)).transpose(0, 3, 1, 2))

def load_image(path):
    return np.array(Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR))[None]

def predict(model_path, batch, topk=3):
    ds = dataset_of(model_path); classes = json.load(open(os.path.join(HERE, "results", f"classes_{ds}.json")))
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    z = sess.run(None, {sess.get_inputs()[0].name: normalize(batch, ds)})[0]
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return [[(classes[j], float(p[i, j])) for j in np.argsort(-p[i])[:topk]] for i in range(len(p))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--model"); ap.add_argument("--test-cifar", action="store_true")
    a = ap.parse_args()
    models = [a.model] if a.model else sorted(glob.glob(os.path.join(HERE, "onnx", "*.onnx")))
    if a.test_cifar:
        from torchvision.datasets import CIFAR10, CIFAR100
        for m in models:
            ds = dataset_of(m); D = CIFAR10 if ds == "cifar10" else CIFAR100; te = D(os.path.join(HERE, "data"), train=False, download=True)
            sess = ort.InferenceSession(m, providers=["CPUExecutionProvider"]); name = sess.get_inputs()[0].name; c = 0
            for i in range(0, len(te.data), 250):
                c += int((sess.run(None, {name: normalize(te.data[i:i+250], ds)})[0].argmax(1) == np.array(te.targets[i:i+250])).sum())
            print(f"{os.path.basename(m):<34} {ds:<9} test acc = {100*c/len(te.data):.2f}%")
        return
    img = load_image(a.image)
    for m in models:
        print(f"{os.path.basename(m):<34}", [f"{l} ({p:.2f})" for l, p in predict(m, img)[0]])

if __name__ == "__main__":
    main()
