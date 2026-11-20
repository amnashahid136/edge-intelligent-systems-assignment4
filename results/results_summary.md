## Q1: baseline vs distillation (teacher 85.13%)

| student            |   params_M |   baseline_acc |   kd_acc |   kd_gain |
|:-------------------|-----------:|---------------:|---------:|----------:|
| mobilenetv2        |      2.352 |          65.01 |    67.74 |      2.73 |
| squeezenet         |      0.777 |          62.4  |    63.65 |      1.25 |
| efficientnet_lite0 |      3.499 |          66.35 |    68.86 |      2.51 |

## ONNX validation

| model                       | arch               | variant   |   torch_acc |   onnx_acc |   max_logit_diff |   onnx_mb |
|:----------------------------|:-------------------|:----------|------------:|-----------:|-----------------:|----------:|
| baseline_mobilenetv2        | mobilenetv2        | baseline  |       65.01 |      65.04 |                0 |    8.947  |
| kd_mobilenetv2              | mobilenetv2        | kd        |       67.74 |      67.72 |                0 |    8.947  |
| baseline_squeezenet         | squeezenet         | baseline  |       62.4  |      62.37 |                0 |    2.9632 |
| kd_squeezenet               | squeezenet         | kd        |       63.65 |      63.64 |                0 |    2.9632 |
| baseline_efficientnet_lite0 | efficientnet_lite0 | baseline  |       66.35 |      66.28 |                0 |   13.3056 |
| kd_efficientnet_lite0       | efficientnet_lite0 | kd        |       68.86 |      68.85 |                0 |   13.3056 |

## Benchmark

| model                       | task         |   params_M |   onnx_size_mb |   torch_acc |   onnx_acc |   lat_mean_ms |   lat_median_ms |   lat_p95_ms |   throughput_ips |   peak_mem_mb |
|:----------------------------|:-------------|-----------:|---------------:|------------:|-----------:|--------------:|----------------:|-------------:|-----------------:|--------------:|
| baseline_mobilenetv2        | Q1 CIFAR-100 |      2.352 |          8.947 |       65.01 |      65.04 |         3.176 |           3.147 |        3.382 |          314.825 |        40.238 |
| kd_mobilenetv2              | Q1 CIFAR-100 |      2.352 |          8.947 |       67.74 |      67.72 |         3.131 |           3.108 |        3.32  |          319.383 |        41.848 |
| baseline_squeezenet         | Q1 CIFAR-100 |      0.777 |          2.963 |       62.4  |      62.37 |         2.319 |           2.111 |        3.24  |          431.189 |        25.98  |
| kd_squeezenet               | Q1 CIFAR-100 |      0.777 |          2.963 |       63.65 |      63.64 |         2.058 |           2.039 |        2.199 |          485.879 |        24.469 |
| baseline_efficientnet_lite0 | Q1 CIFAR-100 |      3.499 |         13.306 |       66.35 |      66.28 |         3.815 |           3.792 |        4.027 |          262.106 |        50.855 |
| kd_efficientnet_lite0       | Q1 CIFAR-100 |      3.499 |         13.306 |       68.86 |      68.85 |         3.853 |           3.82  |        4.07  |          259.51  |        50.387 |
| nas_best                    | Q2 CIFAR-10  |      0.096 |          0.368 |       88    |      88.01 |         0.695 |           0.666 |        0.885 |         1439.69  |        15.648 |
