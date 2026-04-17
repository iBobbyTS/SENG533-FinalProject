from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_SEED = 533

BENCHMARK_ORDER = [
    "mmlu_pro",
    "gsm8k",
    "mbpp",
    "vqa",
    "longbench",
    "truthfulqa",
]

RECOMMENDED_BENCHMARK_CONTEXTS = {
    "mmlu_pro": 8192,
    "gsm8k": 4096,
    "mbpp": 4096,
    "vqa": 4096,
    "longbench": 65536,
    "truthfulqa": 4096,
}

PROFILES = {
    "probe": {
        "mmlu_pro": {"per_category": 1, "cot_shots": 1, "category_limit": 1},
        "gsm8k": {"limit": 1},
        "mbpp": {"limit": 1},
        "vqa": {"limit": 1},
        "longbench": {"per_task": 1, "task_limit": 1},
        "truthfulqa": {"limit": 1},
    },
    "smoke": {
        "mmlu_pro": {"per_category": 1, "cot_shots": 1},
        "gsm8k": {"limit": 10},
        "mbpp": {"limit": 5},
        "vqa": {"limit": 3},
        "longbench": {"per_task": 1},
        "truthfulqa": {"limit": 10},
    },
    "initial": {
        "mmlu_pro": {"per_category": 3, "cot_shots": 3},
        "gsm8k": {"limit": 100},
        "mbpp": {"limit": 25},
        "vqa": {"limit": 30},
        "longbench": {"per_task": 2},
        "truthfulqa": {"limit": 100},
    },
    "mixed": {
        "mmlu_pro": {"per_category": 20, "cot_shots": 5},
        "gsm8k": {"limit": None},
        "mbpp": {"limit": 500},
        "vqa": {"limit": 1000},
        "longbench": {"per_task": 10},
        "truthfulqa": {"limit": None},
    },
}

EXTERNAL_REPOS = {
    "mmlu_pro": {
        "url": "https://github.com/TIGER-AI-Lab/MMLU-Pro.git",
        "dir": "MMLU-Pro",
    },
    "gsm8k": {
        "url": "https://github.com/openai/grade-school-math.git",
        "dir": "grade-school-math",
    },
    "mbpp": {
        "url": "https://github.com/bigcode-project/bigcode-evaluation-harness.git",
        "dir": "bigcode-evaluation-harness",
    },
    "vqa": {
        "url": "https://github.com/GT-Vision-Lab/VQA.git",
        "dir": "VQA",
    },
    "longbench": {
        "url": "https://github.com/THUDM/LongBench.git",
        "dir": "LongBench",
    },
    "truthfulqa": {
        "url": "https://github.com/sylinrl/TruthfulQA.git",
        "dir": "TruthfulQA",
    },
}

OFFICIAL_RESOURCES = {
    "mmlu_pro": {
        "repo": "https://github.com/TIGER-AI-Lab/MMLU-Pro",
        "dataset": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "metric": "multiple-choice accuracy",
        "mode": "adapted",
    },
    "gsm8k": {
        "repo": "https://github.com/openai/grade-school-math",
        "dataset": "https://github.com/openai/grade-school-math/tree/master/grade_school_math/data",
        "metric": "exact match on final numeric answer",
        "mode": "adapted",
    },
    "mbpp": {
        "repo": "https://github.com/bigcode-project/bigcode-evaluation-harness",
        "dataset": "https://huggingface.co/datasets/google-research-datasets/mbpp",
        "metric": "single-sample execution pass rate",
        "mode": "adapted",
    },
    "vqa": {
        "repo": "https://github.com/GT-Vision-Lab/VQA",
        "dataset": "https://visualqa.org/download.html",
        "metric": "open-ended VQA accuracy formula",
        "mode": "adapted",
    },
    "longbench": {
        "repo": "https://github.com/THUDM/LongBench",
        "dataset": "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip",
        "metric": "task-specific official-style metrics",
        "mode": "adapted",
    },
    "truthfulqa": {
        "repo": "https://github.com/sylinrl/TruthfulQA",
        "dataset": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "metric": "binary multiple-choice accuracy",
        "mode": "adapted",
    },
}

VQA_DATA_URLS = {
    "questions": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "image_base": "https://images.cocodataset.org/val2014",
}

LONG_BENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

LONG_BENCH_DATASET_TO_METRIC = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_count": "classification",
    "passage_retrieval_en": "classification",
    "passage_retrieval_zh": "classification",
    "lcc": "edit_similarity",
    "repobench-p": "edit_similarity",
}

LONG_BENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

LONG_BENCH_FALLBACK_PROMPTS = {
    "default": "Read the context and answer the instruction as accurately as possible.\n\nContext:\n{context}\n\nInstruction:\n{input}\n\nAnswer:",
    "lcc": "Please complete the code given below. Only output the completion for the missing next line.\n\n{context}\n\nNext line:",
    "repobench-p": "Use the repository context below to predict the next line of code. Only output the next line.\n\n{context}\n\nNext line:",
}

LONG_BENCH_DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"
CACHE_ROOT = PROJECT_ROOT / "cache"
EXTERNAL_ROOT = PROJECT_ROOT / "external"
