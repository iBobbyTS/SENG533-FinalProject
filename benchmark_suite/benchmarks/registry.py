from __future__ import annotations

from . import gsm8k, longbench, mbpp, mmlu_pro, truthfulqa, vqa

BENCHMARK_REGISTRY = {
    "mmlu_pro": mmlu_pro.run,
    "gsm8k": gsm8k.run,
    "mbpp": mbpp.run,
    "vqa": vqa.run,
    "longbench": longbench.run,
    "truthfulqa": truthfulqa.run,
}
