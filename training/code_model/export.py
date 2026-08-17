"""Production export (Phase 10).

The LoRA/QLoRA adapter `training/code_model/hf_backend.py` saves at the end
of a completed run (`adapter_config.json` + `adapter_model.safetensors`) IS
already the primary, minimum production export -- nothing further is
required to use it (load the base model + `PeftModel.from_pretrained`).

This module adds the two next steps Phase 10 asks for:

1. `merge_adapter_into_base` -- a REAL merge using `peft`'s
   `merge_and_unload()`, producing a standalone full-precision model
   directory loadable with plain `AutoModelForCausalLM.from_pretrained`
   (no `peft` needed at inference time).
2. `gguf_conversion_command` -- this repository has no existing local LLM
   inference runtime to plug a merged model into (JARVIS's voice/TTS
   modules are unrelated), and GGUF conversion requires llama.cpp's own
   external `convert_hf_to_gguf.py` tooling, which this repo does not (and
   should not) vendor. Rather than reimplementing GGUF conversion, this
   documents/returns the exact command to run against a real llama.cpp
   checkout, per Phase 10's explicit "wrap/document it clearly" allowance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MergeResult:
    success: bool
    merged_model_path: str | None
    error: str | None = None


def merge_adapter_into_base(
    adapter_path: str | Path, *, output_path: str | Path, base_model_id: str | None = None,
    trust_remote_code: bool = False,
) -> MergeResult:
    """Loads the base model referenced by the adapter's own
    `adapter_config.json` (or `base_model_id` if given, e.g. to override a
    local path), applies the LoRA weights, calls `merge_and_unload()`
    (peft's real weight-merge, not a placeholder), and saves the result as
    a standalone model directory. Never raises for an ordinary failure --
    returns `MergeResult(success=False, error=...)`.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return MergeResult(False, None, f"required ML packages not installed: {exc}")

    adapter_dir = Path(adapter_path)
    try:
        import json
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        model_id = base_model_id or adapter_config.get("base_model_name_or_path")
        if not model_id:
            return MergeResult(False, None, "could not determine base model id from adapter_config.json")

        base_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged = model.merge_and_unload()

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_dir))

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(out_dir))

        return MergeResult(True, str(out_dir))
    except Exception as exc:
        return MergeResult(False, None, f"{type(exc).__name__}: {exc}")


def _resolve_adapter_path(model_version: str | None, adapter_path: str | None) -> str:
    if adapter_path:
        return adapter_path
    if model_version:
        from brain.learning_training import get_model_registry
        record = get_model_registry().get(model_version)
        if record and record.adapter_path:
            return record.adapter_path
        raise ValueError(f"no adapter_path recorded for model_version {model_version!r} in the model registry")
    raise ValueError("either --model or --adapter must be given")


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Export/merge a trained LoRA/QLoRA adapter into a standalone model (Phase 25).")
    parser.add_argument("--model", default=None, help="model_version to export (looked up in the model registry for its adapter path)")
    parser.add_argument("--adapter", default=None, help="direct path to a saved PEFT adapter directory (alternative to --model)")
    parser.add_argument("--output", required=True, help="output directory for the merged standalone model")
    parser.add_argument("--base-model", default=None, help="override the base model id recorded in the adapter's own config")
    return parser


def _cli_main(argv: list[str] | None = None) -> int:
    import json
    args = build_parser().parse_args(argv)
    try:
        adapter_path = _resolve_adapter_path(args.model, args.adapter)
    except ValueError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    result = merge_adapter_into_base(adapter_path, output_path=args.output, base_model_id=args.base_model)
    print(json.dumps({"success": result.success, "merged_model_path": result.merged_model_path, "error": result.error}, indent=2))
    return 0 if result.success else 1


def gguf_conversion_command(merged_model_path: str | Path, *, llama_cpp_dir: str = "<path-to-llama.cpp-checkout>", output_gguf: str | None = None) -> str:
    """Returns the exact command to run for GGUF conversion -- this repo
    never vendors or shells out to llama.cpp itself; the caller runs this
    in their own llama.cpp checkout (`pip install -r requirements.txt`
    inside it first, per llama.cpp's own setup instructions)."""
    merged = Path(merged_model_path)
    out = output_gguf or str(merged / f"{merged.name}.gguf")
    return (
        f"python {llama_cpp_dir}/convert_hf_to_gguf.py {merged} --outfile {out} --outtype f16\n"
        f"# then, optionally, quantize for local inference (e.g. Q4_K_M):\n"
        f"{llama_cpp_dir}/llama-quantize {out} {out.replace('.gguf', '.Q4_K_M.gguf')} Q4_K_M"
    )


if __name__ == "__main__":
    raise SystemExit(_cli_main())
