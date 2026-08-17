"""CodingAgent-protocol adapter for JARVIS's own trained student model
(Phase 11).

Implements `brain.improvement_coding_agent.CodingAgent`'s exact protocol
(`provider_name`, `.run(task, constraints) -> CodingAgentResult`) so the
trained model plugs into the SAME worktree-isolated harness the Claude
teacher already uses (`brain/improvement_orchestrator.py`,
`training/code_model/benchmark/runner.py`) -- no second tool-safety or
worktree implementation.

Drives a real, bounded inspect -> search -> read -> patch -> test -> revise
loop:
  1. inspect/search/read: `training.code_model.context_packer` builds a
     compact repository context from the task text.
  2. patch: the loaded model (base model + optional trained LoRA adapter)
     is prompted for a structured, parseable patch response -- the same
     `FILE: <path>` / full-file-content format used elsewhere in this
     package -- and it's applied as real file writes.
  3. test: `brain.task_supervisor.SafeCommandRunner` runs pytest for real
     inside the workspace.
  4. revise: on failure, the test output is fed back for up to
     `max_iterations` rounds.

Deliberately simpler than Claude Code's full tool-use loop (no free-form
shell/browse tools) -- a real, working, bounded patch-apply-test loop, not
a stub returning chat text.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from brain.improvement_coding_agent import CodingAgentConstraints, CodingAgentResult
from brain.task_supervisor import SafeCommandRunner
from training.code_model.context_packer import pack_repository_context

_FILE_BLOCK_RE = re.compile(r"FILE:\s*(?P<path>\S+)\s*\n<<<CONTENT>>>\n(?P<content>.*?)\n<<<END>>>", re.DOTALL)

_PROMPT_TEMPLATE = (
    "You are a careful software engineer fixing a real bug in a codebase. "
    "You are given the task and relevant repository evidence. Respond with "
    "the smallest patch that fixes the problem, grounded only in the "
    "evidence given -- never invent files, behavior, or reasoning not "
    "shown to you.\n\n"
    "TASK: {task}\n\n"
    "REPOSITORY STATE:\n{context}\n\n"
    "{feedback}"
    "Respond with ONLY one or more blocks in this exact format, one per "
    "changed file, with the file's COMPLETE new content:\n"
    "FILE: <relative/path.py>\n<<<CONTENT>>>\n<full new file content>\n<<<END>>>\n"
)


def parse_patch_response(text: str) -> dict[str, str]:
    """Real, deterministic parsing of the model's structured patch
    response -- never free-form guessing. Returns `{}` (not an error) if
    the model's response didn't include a recognizable block; the caller
    treats that as "no usable patch this round," feeding it back as
    corrective context."""
    return {m.group("path").strip(): m.group("content") for m in _FILE_BLOCK_RE.finditer(text)}


class LocalCodingModelAdapter:
    """`CodingAgent` implementation backed by a real, loaded HF causal-LM
    (optionally with a trained LoRA adapter applied). Construct directly
    with an already-loaded `model`/`tokenizer` pair, or via
    `from_checkpoint(...)` to load them from a base model id + optional
    adapter path."""

    provider_name = "local_student_model"

    def __init__(self, model, tokenizer, *, max_iterations: int = 2, max_new_tokens: int = 512, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.max_new_tokens = max_new_tokens
        self.device = device or "cpu"

    @classmethod
    def from_checkpoint(
        cls, base_model_id: str, adapter_path: str | None = None, *,
        trust_remote_code: bool = False, max_iterations: int = 2, max_new_tokens: int = 512,
    ) -> "LocalCodingModelAdapter":
        """Real model loading -- `AutoModelForCausalLM.from_pretrained` for
        the base model, `PeftModel.from_pretrained` for the trained adapter
        when one is given (an untrained/base-only student passes
        `adapter_path=None`, matching Phase 16's "evaluate the original/
        base open coding model" comparison arm)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return cls(model, tokenizer, device=device, max_iterations=max_iterations, max_new_tokens=max_new_tokens)

    def _generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def run(self, task: str, constraints: CodingAgentConstraints) -> CodingAgentResult:
        started = time.time()
        workspace = Path(constraints.workspace)
        runner = SafeCommandRunner()
        feedback = ""
        model_calls = 0
        model_name = getattr(self.model, "name_or_path", None) or getattr(getattr(self.model, "base_model", None), "name_or_path", None)

        try:
            for iteration in range(1, self.max_iterations + 1):
                context = pack_repository_context(workspace, keywords=[w for w in task.split() if len(w) > 3][:10])
                prompt = _PROMPT_TEMPLATE.format(task=task, context=context.render(), feedback=feedback)
                response_text = self._generate(prompt)
                model_calls += 1

                patch = parse_patch_response(response_text)
                if not patch:
                    feedback = "Your previous response did not include a parseable FILE:/<<<CONTENT>>> block. Try again.\n\n"
                    continue

                for relative_path, content in patch.items():
                    target = workspace / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

                try:
                    test_result = runner.run(["python", "-m", "pytest", "-q"], str(workspace), timeout=min(120.0, constraints.timeout_seconds))
                except Exception as exc:
                    test_result = {"exit_code": -1, "output": str(exc)}

                if test_result.get("exit_code") == 0:
                    return CodingAgentResult(
                        exit_status="completed", provider=self.provider_name, model=model_name,
                        model_calls=model_calls, started_at=started, ended_at=time.time(),
                        stdout_summary=f"tests passed after {iteration} iteration(s)",
                    )
                feedback = f"Your previous patch did not pass tests. Test output:\n{str(test_result.get('output', ''))[-2000:]}\n\n"

            return CodingAgentResult(
                exit_status="completed", provider=self.provider_name, model=model_name,
                model_calls=model_calls, started_at=started, ended_at=time.time(),
                stdout_summary="max iterations reached without passing tests", error="max_iterations_exhausted",
            )
        except Exception as exc:
            return CodingAgentResult(
                exit_status="crashed", provider=self.provider_name, model=model_name,
                started_at=started, ended_at=time.time(), error=f"{type(exc).__name__}: {exc}",
            )
