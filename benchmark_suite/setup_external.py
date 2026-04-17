from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset

from .config import (
    EXTERNAL_REPOS,
    LONG_BENCH_DATA_URL,
    OFFICIAL_RESOURCES,
    VQA_DATA_URLS,
)
from .utils import ensure_dir


def configure_hf_cache(root: Path) -> None:
    hf_home = root / "cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    ensure_dir(hf_home)
    ensure_dir(hf_home / "hub")
    ensure_dir(hf_home / "datasets")


def clone_repo(url: str, destination: Path) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    urlretrieve(url, destination)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def setup_mmlu_pro(root: Path) -> None:
    repo = EXTERNAL_REPOS["mmlu_pro"]
    clone_repo(repo["url"], root / "external" / repo["dir"])
    load_dataset("TIGER-Lab/MMLU-Pro", split="test[:1]", cache_dir=str(root / "cache" / "hf"))


def setup_gsm8k(root: Path) -> None:
    repo = EXTERNAL_REPOS["gsm8k"]
    clone_repo(repo["url"], root / "external" / repo["dir"])


def setup_mbpp(root: Path) -> None:
    repo = EXTERNAL_REPOS["mbpp"]
    clone_repo(repo["url"], root / "external" / repo["dir"])
    load_dataset("google-research-datasets/mbpp", "full", split="test[:1]", cache_dir=str(root / "cache" / "hf"))


def setup_vqa(root: Path) -> None:
    repo = EXTERNAL_REPOS["vqa"]
    clone_repo(repo["url"], root / "external" / repo["dir"])
    data_dir = root / "data" / "vqa_v2"
    ensure_dir(data_dir)
    questions_zip = data_dir / "v2_Questions_Val_mscoco.zip"
    annotations_zip = data_dir / "v2_Annotations_Val_mscoco.zip"
    download_file(VQA_DATA_URLS["questions"], questions_zip)
    download_file(VQA_DATA_URLS["annotations"], annotations_zip)
    questions_json = data_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_json = data_dir / "v2_mscoco_val2014_annotations.json"
    if not questions_json.exists():
        extract_zip(questions_zip, data_dir)
    if not annotations_json.exists():
        extract_zip(annotations_zip, data_dir)


def setup_longbench(root: Path) -> None:
    repo = EXTERNAL_REPOS["longbench"]
    clone_repo(repo["url"], root / "external" / repo["dir"])
    data_dir = root / "data" / "longbench"
    data_zip = data_dir / "data.zip"
    download_file(LONG_BENCH_DATA_URL, data_zip)
    if not list(data_dir.rglob("narrativeqa.jsonl")):
        extract_zip(data_zip, data_dir)


def setup_truthfulqa(root: Path) -> None:
    repo = EXTERNAL_REPOS["truthfulqa"]
    clone_repo(repo["url"], root / "external" / repo["dir"])


SETUP_HANDLERS = {
    "mmlu_pro": setup_mmlu_pro,
    "gsm8k": setup_gsm8k,
    "mbpp": setup_mbpp,
    "vqa": setup_vqa,
    "longbench": setup_longbench,
    "truthfulqa": setup_truthfulqa,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark test environments.")
    parser.add_argument("--bench", choices=list(SETUP_HANDLERS) + ["all"], required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_hf_cache(args.root)
    if args.bench == "all":
        for name in SETUP_HANDLERS:
            print(f"[setup] {name}: {OFFICIAL_RESOURCES[name]['repo']}")
            SETUP_HANDLERS[name](args.root)
        return 0
    print(f"[setup] {args.bench}: {OFFICIAL_RESOURCES[args.bench]['repo']}")
    SETUP_HANDLERS[args.bench](args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
