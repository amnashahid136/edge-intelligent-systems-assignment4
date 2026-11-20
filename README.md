# Assignment 4 - Edge Computing & Intelligent Systems  (21i-1713_A)

Q1: Knowledge distillation (ResNet-50 teacher -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0, CIFAR-100) + ONNX Runtime CPU inference.
Q2: Hardware-aware NAS with Optuna (CIFAR-10) measured on real CPU latency / size / memory + ONNX Runtime CPU inference.

## Requirements
Python >= 3.10, PyTorch, torchvision, onnx, onnxruntime, onnxscript, optuna, numpy, pandas, matplotlib, pillow, psutil.
`pip install torch torchvision onnx onnxruntime onnxscript optuna numpy pandas matplotlib pillow psutil`
Training used a Kaggle GPU runtime (T4, fp16 mixed precision). The notebook falls back to CPU automatically (very slow).

## How to run (training + everything)
1. Import `Assignment4_Edge_KD_NAS_Kaggle.ipynb` into Kaggle; Session options: Accelerator = GPU, Internet = On.
2. `Run All`. CIFAR-10/100 and ImageNet-pretrained ResNet-50 weights are downloaded by the notebook.
   Set `QUICK = True` in the configuration cell for a few-minute smoke test.
3. Outputs are written to `assignment4_outputs/` (layout below).

## How to run inference only (no training, CPU)
```
cd assignment4_outputs
python infer_onnx.py --image samples/<any image>.png                 # all 7 ONNX models
python infer_onnx.py --image my.jpg --model onnx/kd_mobilenetv2.onnx # one model
python infer_onnx.py --test-cifar                                    # test-set accuracy of every model through ONNX Runtime
python benchmark_onnx.py onnx/nas_best.onnx --runs 200 --threads 1   # latency / throughput / peak memory
```

## Layout
| Path | Deliverable |
|---|---|
| `checkpoints/teacher_resnet50.pt` | 1. Teacher checkpoint (training script = notebook sec. 4) |
| notebook sec. 3 | 2. Student architectures (MobileNetV2, SqueezeNet, EfficientNet-Lite0) |
| `checkpoints/baseline_*.pt` | 3. Baseline students (sec. 5) |
| `checkpoints/kd_*.pt` | 4. Distilled students + KD implementation (sec. 6) |
| notebook sec. 10 | 5. NAS search space + architecture generator |
| notebook sec. 12 | 6. Optuna (TPE) search |
| `results/nas_search_history.csv/json`, `results/best_architecture.json` | 7. NAS results |
| `checkpoints/nas_best.pt` | 8. Optimised architecture checkpoint |
| `infer_onnx.py`, notebook sec. 8 (`InferencePipeline`) | 9. Inference pipeline |
| `onnx/*.onnx` | 10. ONNX models (+ ONNX Runtime CPU scripts) |
| `benchmark_onnx.py`, `results/benchmark_results.csv` | 11. Benchmarking |
| `figures/*.png`, `results/*.csv` | material for the technical report (12) |

## Key results
| model                       | task         |   params_M |   onnx_size_mb |   torch_acc |   onnx_acc |   lat_mean_ms |   lat_median_ms |   lat_p95_ms |   throughput_ips |   peak_mem_mb |
|:----------------------------|:-------------|-----------:|---------------:|------------:|-----------:|--------------:|----------------:|-------------:|-----------------:|--------------:|
| baseline_mobilenetv2        | Q1 CIFAR-100 |      2.352 |          8.947 |       65.01 |      65.04 |         3.176 |           3.147 |        3.382 |          314.825 |        40.238 |
| kd_mobilenetv2              | Q1 CIFAR-100 |      2.352 |          8.947 |       67.74 |      67.72 |         3.131 |           3.108 |        3.32  |          319.383 |        41.848 |
| baseline_squeezenet         | Q1 CIFAR-100 |      0.777 |          2.963 |       62.4  |      62.37 |         2.319 |           2.111 |        3.24  |          431.189 |        25.98  |
| kd_squeezenet               | Q1 CIFAR-100 |      0.777 |          2.963 |       63.65 |      63.64 |         2.058 |           2.039 |        2.199 |          485.879 |        24.469 |
| baseline_efficientnet_lite0 | Q1 CIFAR-100 |      3.499 |         13.306 |       66.35 |      66.28 |         3.815 |           3.792 |        4.027 |          262.106 |        50.855 |
| kd_efficientnet_lite0       | Q1 CIFAR-100 |      3.499 |         13.306 |       68.86 |      68.85 |         3.853 |           3.82  |        4.07  |          259.51  |        50.387 |
| nas_best                    | Q2 CIFAR-10  |      0.096 |          0.368 |       88    |      88.01 |         0.695 |           0.666 |        0.885 |         1439.69  |        15.648 |

Teacher ResNet-50 (CIFAR-100): 85.13%.   KD hyper-parameters: T=4.0, alpha=0.7, beta(feature)=1.0.
NAS constraints: latency <= 10.0 ms, size <= 2.0 MB, peak memory <= 150.0 MB. Best architecture: res32 stem24 | [1x dws k3 c48 silu] > [3x dws k7 c128 hardswish] > [3x conv k3 c24 relu]
