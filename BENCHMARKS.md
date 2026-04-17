# LM Studio Benchmark Suite

The benchmark scripts in this directory are used to validate basic local execution of `qwen3.5-0.8b` against several public benchmarks through LM Studio.

## Layout

- `scripts/setup_*.sh`: download the benchmark-specific environment or reference repo
- `scripts/run_*.sh`: run a single benchmark
- `scripts/run_all_benchmarks.sh`: run the full benchmark suite
- `scripts/list_loaded_models.sh`: list currently loaded LM Studio models
- `scripts/load_model.sh`: load a target model through the LM Studio native API
- `scripts/unload_all_models.sh`: unload all currently loaded LM Studio models
- `scripts/unload_model.sh`: unload a specific model instance by instance ID
- `benchmark_suite/`: shared Python implementation
- `results/`: output artifacts

## Supported Benchmarks

- `mmlu_pro`
- `gsm8k`
- `mbpp`
- `vqa`
- `longbench`
- `truthfulqa`

## Profile

- `probe`: single-sample path validation, useful for large-model first checks or quick pipeline verification
- `smoke`: minimal end-to-end validation, useful for confirming the scripts work
- `initial`: early-stage validation, useful for the current round of quick comparisons
- `mixed`: larger mixed-scale runs based on the earlier evaluation plan

## Default Assumptions

- The LM Studio service is running at `http://127.0.0.1:1234/v1`
- The default model is `qwen3.5-0.8b`
- If the model overthinks, loops, or gets truncated by the context window, the scripts record that behavior instead of treating it as a framework error
- Single-benchmark runs list and unload any currently loaded models first, then load the target model, and unload it again after completion
- Full-suite runs perform model management only once at the outermost level, while inner benchmark invocations explicitly skip duplicate model-management steps

## Monitoring Scripts

- `scripts/run_monitored_benchmark.sh`: monitored single-benchmark runner; the script starts `macmon`, records power, samples LM Studio memory and virtual memory, and produces a per-benchmark monitoring summary
- `scripts/finalize_monitored_run.sh`: combine per-benchmark benchmark, memory, and power results
- `scripts/aggregate_monitored_runs.sh`: aggregate multiple monitored rows into a Markdown table

Memory monitoring records the LM Studio backend `node` process with the largest RSS by default, including RSS, VMS, and page-in counters. It also samples system swap usage via `sysctl vm.swapusage`. If process RSS is unavailable, the summary falls back to `macmon`'s `memory.ram_usage` for physical memory, and to `macmon`'s `memory.swap_usage` for swap if available.
Power monitoring is enabled by default. If `macmon` cannot be started in the current environment, append `--disable-power-monitoring` to `scripts/run_monitored_benchmark.sh` or `scripts/run_all_benchmarks.sh` to disable it explicitly.
