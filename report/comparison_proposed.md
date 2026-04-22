# Intelligence comparison
1. Model size (All Q4): 4B-Q4_K_M, 9B-Q4_K_M, 27B-Q4_K_M
2. Dense vs MoE: 27B-Q4_K_M, 35B-A3B-Q4_K_M
3. All quantization levels: 9B-Q4_K_M, 9B-Q6_K, 9B-Q8_0
4. Affect of quantization on different model size: 9B-Q4_K_M, 9B-Q6_K, 27B-Q4_K_M, 27B-Q6_K_M
5. Distillation: 27B-Q4_K_M, 27B-Q4_K_M-Opus4.6
6. Ultra low quantization: 27B-Q4_K_M-Opus4.6, 27B-Q2_K-Opus4.6

# Memory usage

Context mappings used in the benchmark runner:
- `ctx_4096`: `gsm8k`, `mbpp`, `vqa`, `truthfulqa`
- `ctx_8192`: `mmlu_pro`
- `ctx_65536`: `longbench`
- `ctx_max_262144`: Qwen3.5 maximum supported context length

All values below were estimated with `lms load --estimate-only` against the LM Studio registry.

| Params  | Quantization | Size on Disk | Notes                             | Load with `ctx_4096` | Load with `ctx_8192` | Load with `ctx_65536` | Load with `ctx_max_262144` |
|---------|--------------|-------------:|-----------------------------------|---------------------:|---------------------:|----------------------:|---------------------------:|
| 4B      | Q4_K_M       |     3.15 GiB | Base                              |             3.62 GiB |             3.76 GiB |              5.65 GiB |                  12.13 GiB |
| 9B      | Q4_K_M       |     6.10 GiB | Base, current selected variant    |             6.95 GiB |             7.16 GiB |             10.04 GiB |                  19.92 GiB |
| 9B      | Q6_K         |     7.71 GiB | Base, previously selected variant |             8.77 GiB |             9.02 GiB |             12.44 GiB |                  24.18 GiB |
| 9B      | Q8_0         |     9.73 GiB | Base, previously selected variant |            11.05 GiB |            11.35 GiB |             15.45 GiB |                  29.51 GiB |
| 27B     | Q4_K_M       |    16.27 GiB | Base                              |            18.47 GiB |            18.95 GiB |             25.65 GiB |                  48.65 GiB |
| 27B     | Q6_K         |    21.43 GiB | Base                              |            24.29 GiB |            24.90 GiB |             33.34 GiB |                  62.28 GiB |
| 27B     | Q2_K         |    10.29 GiB | Opus 4.6 distilled                |            11.72 GiB |            12.05 GiB |             16.75 GiB |                  32.87 GiB |
| 27B     | Q4_K_M       |    16.27 GiB | Opus 4.6 distilled                |            18.47 GiB |            18.95 GiB |             25.65 GiB |                  48.65 GiB |
| 35B-A3B | Q4_K_M       |    20.56 GiB | Official A3B MoE                  |            21.31 GiB |            21.44 GiB |             23.30 GiB |                  29.66 GiB |
