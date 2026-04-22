# LM Studio Benchmark Commands

This README lists the commands used to prepare the benchmark environment, download public benchmark datasets, and run each test group in this repository.

## Prerequisites

- LM Studio server is running. The scripts use `http://127.0.0.1:1234/v1` by default.
- Python 3 is available on the host machine.
- `scripts/setup_all.sh` creates `.venv-bench`, installs Python dependencies, and downloads the public benchmark assets.

## Download Benchmark Datasets

Download everything:

```bash
bash scripts/setup_all.sh
```

Download a single benchmark dataset:

```bash
bash scripts/setup_gsm8k.sh
bash scripts/setup_mmlu_pro.sh
bash scripts/setup_mbpp.sh
bash scripts/setup_vqa.sh
bash scripts/setup_longbench.sh
bash scripts/setup_truthfulqa.sh
```

## Run Capability Benchmarks

Run the full benchmark suite with the `initial` profile:

```bash
bash scripts/run_all_benchmarks.sh initial --model <model>
```

Run a single benchmark:

```bash
bash scripts/run_gsm8k.sh initial --model <model>
bash scripts/run_mmlu_pro.sh initial --model <model>
bash scripts/run_mbpp.sh initial --model <model>
bash scripts/run_vqa.sh initial --model <model>
bash scripts/run_longbench.sh initial --model <model>
bash scripts/run_truthfulqa.sh initial --model <model>
```

Available profiles:

- `probe`: single-sample pipeline check
- `smoke`: minimal end-to-end validation
- `initial`: quick comparison profile
- `mixed`: larger mixed-scale evaluation profile

Optional monitored single-benchmark run:

```bash
bash scripts/run_monitored_benchmark.sh mmlu_pro initial --model <model>
```

## Run Input Processing Latency (TTFT)

```bash
python3 scripts/run_input_ttft_confidence_interval.py \
  --models <model> \
  --input-dir <path-to-input-files> \
  --context-length 20000 \
  --runs 5 \
  --max-output-tokens 16
```

Optional overrides:

```bash
python3 scripts/run_input_ttft_confidence_interval.py \
  --base-url http://192.168.31.76:1234/v1 \
  --models <model> \
  --input-dir <path-to-input-files> \
  --context-length 20000 \
  --runs 5 \
  --max-output-tokens 16
```

## Run Output Generation Throughput

```bash
python3 scripts/run_stream_perf_confidence_interval.py \
  --models <model> \
  --contexts 32768 \
  --runs 5 \
  --max-output-tokens 32768 \
  --temperature 0.0
```

If a model does not produce a stable long output at `0.0`, rerun with a slightly higher temperature:

```bash
python3 scripts/run_stream_perf_confidence_interval.py \
  --models <model> \
  --contexts 32768 \
  --runs 5 \
  --max-output-tokens 32768 \
  --temperature 0.1
```

## Run Power Tests

```bash
python3 scripts/run_power_memory_tests.py \
  --models <model> \
  --mode power \
  --context-length 32768 \
  --max-output-tokens 32768 \
  --temperature 0.1 \
  --power-runs 5
```

## Run Memory Tests

```bash
python3 scripts/run_power_memory_tests.py \
  --models <model> \
  --mode memory \
  --context-length 32768 \
  --max-output-tokens 32768 \
  --temperature 0.1
```

Run both power and memory in one pass:

```bash
python3 scripts/run_power_memory_tests.py \
  --models <model> \
  --mode both \
  --context-length 32768 \
  --max-output-tokens 32768 \
  --temperature 0.1 \
  --power-runs 5
```

## Results

All generated artifacts are written under `results/`, usually in a run-specific subdirectory such as `results/<run-id>/`.

For additional script behavior and benchmark layout details, see [`BENCHMARKS.md`](./BENCHMARKS.md).
