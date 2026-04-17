from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset

from ..config import OFFICIAL_RESOURCES, PROFILES
from ..types import RunConfig
from ..utils import apply_model_prompt_suffix, detect_repetition, ensure_dir, sample_list, write_jsonl

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _execute(code: str, test_setup_code: str, tests: list[str], scratch_root: Path) -> tuple[bool, str]:
    ensure_dir(scratch_root)
    with tempfile.TemporaryDirectory(dir=scratch_root) as temp_dir:
        workdir = Path(temp_dir).resolve()
        script_path = workdir / "solution.py"
        body = [code, "", test_setup_code or "", ""]
        body.extend(tests)
        script_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{str(workdir)}:/work",
            "-w",
            "/work",
            "python:3.11-slim",
            "python",
            "/work/solution.py",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Execution timed out after 30 seconds."


def run(config: RunConfig, client) -> dict:
    benchmark_dir = ensure_dir(config.output_root / config.benchmark)
    profile = PROFILES[config.profile]["mbpp"]
    dataset_cache = str(config.cache_root / "hf")
    rows = list(load_dataset("google-research-datasets/mbpp", "full", split="test", cache_dir=dataset_cache))
    sampled = sample_list(rows, profile["limit"], config.seed)
    predictions = []
    correct = 0
    notes = [
        "Uses an adapted official-style MBPP prompt and executes `test_list` inside Docker.",
        "This score is single-sample execution pass rate, not the official pass@k."
    ]
    scratch_root = config.cache_root / "mbpp_exec"

    for index, row in enumerate(sampled):
        prompt = apply_model_prompt_suffix(
            config.model,
            (
            "Use the following MBPP-style prompt to write the solution.\n"
            "Match the function name, parameters, and return format implied by the test exactly.\n"
            "Return only Python code inside one fenced code block. Do not include explanations, example usage, or a main block.\n\n"
            "\"\"\"\n"
            f"{row['text']}\n"
            f"{row['test_list'][0]}\n"
            "\"\"\"\n"
            ),
        )
        response = client.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=config.seed,
            stream=True,
            stream_idle_timeout_seconds=300,
        )
        code = _extract_code(response["content"])
        passed, execution_log = _execute(code, row.get("test_setup_code", ""), list(row["test_list"]), scratch_root)
        correct += int(passed)
        predictions.append(
            {
                "index": index,
                "task_id": row["task_id"],
                "prompt": prompt,
                "correct": passed,
                "finish_reason": response["finish_reason"],
                "loop_risk": detect_repetition(response["content"]),
                "usage": response["usage"],
                "response_text": response["content"],
                "code": code,
                "execution_log": execution_log,
                "stream_debug": response.get("stream_debug"),
            }
        )

    write_jsonl(benchmark_dir / "predictions.jsonl", predictions)
    return {
        "benchmark": config.benchmark,
        "profile": config.profile,
        "status": "completed",
        "model": config.model,
        "base_url": config.base_url,
        "sample_count": len(sampled),
        "metric": OFFICIAL_RESOURCES["mbpp"]["metric"],
        "score": correct / len(sampled) if sampled else None,
        "correct_count": correct,
        "notes": notes,
        "repo": OFFICIAL_RESOURCES["mbpp"]["repo"],
        "dataset": OFFICIAL_RESOURCES["mbpp"]["dataset"],
    }
