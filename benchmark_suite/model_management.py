from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .config import DEFAULT_BASE_URL


def native_api_root_from_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        native_path = path
    elif path.endswith("/v1"):
        native_path = f"{path[:-3]}/api/v1"
    elif path:
        native_path = f"{path}/api/v1"
    else:
        native_path = "/api/v1"
    return urlunparse((parsed.scheme, parsed.netloc, native_path, "", "", ""))


@dataclass
class LoadedInstance:
    model_key: str
    display_name: str
    instance_id: str
    config: dict[str, Any]


class LMStudioModelManager:
    def __init__(self, base_url: str) -> None:
        self.native_api_root = native_api_root_from_base_url(base_url)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.native_api_root}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio model API {method} {url} failed: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LM Studio model API {method} {url} failed: {exc}") from exc
        if not body.strip():
            return {}
        return json.loads(body)

    def list_models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/models")
        return response.get("models", [])

    def list_loaded_instances(self) -> list[LoadedInstance]:
        loaded: list[LoadedInstance] = []
        for model in self.list_models():
            for instance in model.get("loaded_instances", []):
                instance_id = instance.get("instance_id") or instance.get("id") or model.get("key")
                if not instance_id:
                    continue
                loaded.append(
                    LoadedInstance(
                        model_key=model.get("key", ""),
                        display_name=model.get("display_name", model.get("key", "")),
                        instance_id=instance_id,
                        config=instance.get("config", {}),
                    )
                )
        return loaded

    def unload_instance(self, instance_id: str) -> dict[str, Any]:
        return self._request("POST", "/models/unload", {"instance_id": instance_id})

    def unload_all(self) -> list[LoadedInstance]:
        loaded = self.list_loaded_instances()
        for instance in loaded:
            self.unload_instance(instance.instance_id)
        return loaded

    def load_model(self, model: str, context_length: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "echo_load_config": True}
        if context_length is not None:
            payload["context_length"] = context_length
        return self._request("POST", "/models/load", payload)


def format_loaded_instances(instances: list[LoadedInstance]) -> str:
    if not instances:
        return "No loaded models."
    lines = []
    for instance in instances:
        config_bits = []
        context_length = instance.config.get("context_length")
        if context_length is not None:
            config_bits.append(f"context_length={context_length}")
        config_suffix = f" ({', '.join(config_bits)})" if config_bits else ""
        lines.append(f"{instance.instance_id} <- {instance.display_name}{config_suffix}")
    return "\n".join(lines)


@contextmanager
def managed_model(
    base_url: str,
    model: str,
    *,
    context_length: int | None = None,
    enabled: bool = True,
):
    if not enabled:
        yield None
        return

    manager = LMStudioModelManager(base_url)
    preexisting = manager.list_loaded_instances()
    print(format_loaded_instances(preexisting), flush=True)
    if preexisting:
        manager.unload_all()
        print("Unloaded preexisting models.", flush=True)
    load_result = manager.load_model(model, context_length=context_length)
    instance_id = load_result.get("instance_id") or model
    print(f"Loaded model: {instance_id}", flush=True)
    try:
        yield load_result
    finally:
        try:
            manager.unload_instance(instance_id)
            print(f"Unloaded model: {instance_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to unload model {instance_id}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage LM Studio model loading via the native REST API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-loaded", help="List currently loaded models.")
    list_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    list_parser.add_argument("--json", action="store_true")

    unload_all_parser = subparsers.add_parser("unload-all", help="Unload all currently loaded models.")
    unload_all_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    unload_all_parser.add_argument("--json", action="store_true")

    unload_parser = subparsers.add_parser("unload", help="Unload one model instance.")
    unload_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    unload_parser.add_argument("--instance-id", required=True)

    load_parser = subparsers.add_parser("load", help="Load one model.")
    load_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    load_parser.add_argument("--model", required=True)
    load_parser.add_argument("--context-length", type=int, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = LMStudioModelManager(args.base_url)

    if args.command == "list-loaded":
        loaded = manager.list_loaded_instances()
        if args.json:
            print(json.dumps([instance.__dict__ for instance in loaded], ensure_ascii=False, indent=2))
        else:
            print(format_loaded_instances(loaded))
        return 0

    if args.command == "unload-all":
        unloaded = manager.unload_all()
        if args.json:
            print(json.dumps([instance.__dict__ for instance in unloaded], ensure_ascii=False, indent=2))
        else:
            print(format_loaded_instances(unloaded))
        return 0

    if args.command == "unload":
        response = manager.unload_instance(args.instance_id)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    if args.command == "load":
        response = manager.load_model(args.model, context_length=args.context_length)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
