# Memory Metric Reliability

## Question

Which metric is the most reliable for LM Studio memory analysis on macOS, especially for local LLM inference with file-mapped GGUF weights and long context windows?

## Sources

- LM Studio CLI docs: [lms load](https://lmstudio.ai/docs/cli/local-models/load)
- Apple Developer: [Reducing your app's memory use](https://developer.apple.com/documentation/xcode/reducing-your-app-s-memory-use)
- Apple Developer video summary: [Detect and diagnose memory issues](https://developer.apple.com/videos/play/wwdc2021/10180)

Local primary sources also used during validation:
- `man vmmap`
- `man footprint`

## Metrics examined

### 1. `lms load --estimate-only`

Use case:
- Pre-load capacity planning
- Fast comparison across models or context sizes

Strength:
- Directly reflects LM Studio runtime's own estimate
- Useful before actually loading a model

Limitation:
- It is not a live measurement
- It cannot tell how much memory is currently dirty, swapped, or resident

### 2. `ps RSS`

Use case:
- Approximate total resident memory of the LM Studio backend process
- Better than `footprint` when the question is "how much RAM is actually resident for this process right now"

Strength:
- Includes resident mapped-file pages such as GGUF weights
- Easy to sample repeatedly

Limitation:
- Can still include clean resident pages and shared pages
- Does not directly represent macOS memory pressure accounting

### 3. `footprint`

Use case:
- Best metric for process memory pressure on macOS
- Best answer to "what does the OS effectively charge to this process"

Strength:
- `footprint` defines process footprint around dirty memory accounting
- De-duplicates shared objects
- Exposes `phys_footprint` and `phys_footprint_peak`

Limitation:
- It undercounts large file-mapped clean model weights
- Therefore it is not suitable as the only metric for "how much RAM the model occupies in total"

### 4. `memory_analysis.py` / `vmmap -summary`

Use case:
- One-off structural attribution
- Good for answering "which bucket grew: model, context/GPU, or software"

Strength:
- Gives interpretable buckets
- Clearly reveals context growth and swap emergence

Limitation:
- It is a snapshot, not continuous monitoring
- Summing its bucket residents is not a safe proxy for total memory
- It can overcount relative to `phys_footprint`

## Validation A: Same model, different context

Model:
- `qwen/qwen3.5-9b@q4_k_m`

Result directory:
- [20260402T200336Z_memory_analysis_ctx_check](/Users/ibobby/School/SENG%20533/Final%20Project/results/20260402T200336Z_memory_analysis_ctx_check)

Observed `memory_analysis.py` buckets:

| Context | Model bucket | Context/GPU bucket | Software bucket | Swapped |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 5.20 GB | 1.21 GB | 1.04 GB | 0 |
| 8192 | 5.20 GB | 1.31 GB | 1.03 GB | 0 |
| 65536 | 5.20 GB | 3.11 GB | 1.05 GB | 0 |
| 262144 | 5.20 GB | 8.51 GB | 825.74 MB | 748.50 MB |

Conclusion:
- The script is good at showing the relative growth of the context-related bucket.
- It is also useful for detecting when swap starts to appear.
- It should not be used as a total-memory meter.

## Validation B: Different models, same context

Context:
- `65536`

Result directory:
- [20260402T201040Z_memory_metric_validation](/Users/ibobby/School/SENG%20533/Final%20Project/results/20260402T201040Z_memory_metric_validation)

Summary:

| Model | LM Studio estimate | `ps RSS` | `footprint phys_footprint` | `memory_analysis` resident sum |
| --- | ---: | ---: | ---: | ---: |
| 4B Q4_K_M | 5.65 GiB | 5.68 GiB | 3.24 GiB | 6.28 GiB |
| 9B Q4_K_M | 10.04 GiB | 8.64 GiB | 3.49 GiB | 9.29 GiB |
| 27B Q4_K_M | 25.65 GiB | 20.23 GiB | 5.96 GiB | 20.72 GiB |

Interpretation:
- `ps RSS` and `memory_analysis` track model-size growth in a way that matches intuition for local LLM loading.
- `footprint` grows much more slowly because large file-mapped model weights remain mostly clean and therefore are not charged to process footprint.
- For the 27B model, `footprint` explicitly shows a huge `mapped file` clean region of roughly 16.53 GiB, which explains why `phys_footprint` stays far below RSS.

## Recommendation

Use different metrics for different questions:

### A. "Will the model fit before I load it?"

Use:
- `lms load --estimate-only`

### B. "How much resident memory does the LM Studio backend occupy?"

Use:
- `ps RSS` of the largest LM Studio backend `node` process

Also record:
- `sysctl vm.swapusage`

This combination is the most practical answer for local LLM experiments.

### C. "How much memory pressure is the OS charging to the process?"

Use:
- `footprint`
- Especially `phys_footprint` and `phys_footprint_peak`

### D. "Which part grew when I increased context?"

Use:
- `memory_analysis.py`

But only as a structural breakdown, not as a total-memory metric.

## Final judgment

The most reliable overall workflow for this project is:

1. `lms load --estimate-only` for pre-load planning
2. `ps RSS` for total resident backend memory
3. `sysctl vm.swapusage` for swap pressure
4. `footprint phys_footprint` for macOS memory-pressure accounting
5. `memory_analysis.py` only for occasional bucket-level attribution

This avoids the main failure mode of relying on any single metric:
- `footprint` undercounts clean mapped weights
- `RSS` does not explain memory pressure semantics
- `memory_analysis.py` is not a total-memory meter
