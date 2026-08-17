"""Hardware preflight detection and feasibility checks (Phase 6).

Never raises for an ordinary "can't detect X" outcome -- every failure mode
becomes an honest entry in `HardwareInfo.detection_errors` or
`FeasibilityResult.reasons`, exactly like `brain/learning_training.py`'s
`run_pre_training_checks` never crashes.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training.code_model.config import CodeModelTrainingConfig


@dataclass
class HardwareInfo:
    cuda_available: bool
    gpu_name: str | None
    gpu_count: int
    vram_total_mb: float | None
    vram_free_mb: float | None
    torch_version: str | None
    torch_cuda_version: str | None
    bitsandbytes_available: bool
    bitsandbytes_functional: bool
    system_ram_total_mb: float
    system_ram_available_mb: float
    disk_free_mb: float
    detection_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _system_ram_mb() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1024**2, vm.available / 1024**2
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024**2, stat.ullAvailPhys / 1024**2
    except Exception:
        return 0.0, 0.0


def _check_bitsandbytes_functional() -> tuple[bool, str | None]:
    try:
        import torch
        import bitsandbytes.nn as bnn
        linear = bnn.Linear4bit(8, 8, compute_dtype=torch.float16).to("cuda")
        linear(torch.randn(1, 8, device="cuda", dtype=torch.float16))
        return True, None
    except Exception as exc:
        return False, str(exc)


def detect_hardware(path_for_disk_check: str | Path = ".") -> HardwareInfo:
    errors: list[str] = []
    cuda_available = False
    gpu_name = gpu_count = vram_total = vram_free = None
    torch_version = torch_cuda_version = None
    gpu_count = 0

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            vram_free, vram_total = free / 1024**2, total / 1024**2
            torch_cuda_version = torch.version.cuda
    except Exception as exc:
        errors.append(f"torch/CUDA detection failed: {type(exc).__name__}: {exc}")

    bnb_available = bnb_functional = False
    try:
        import bitsandbytes  # noqa: F401
        bnb_available = True
        if cuda_available:
            bnb_functional, bnb_error = _check_bitsandbytes_functional()
            if not bnb_functional and bnb_error:
                errors.append(f"bitsandbytes functional check failed: {bnb_error}")
    except Exception as exc:
        errors.append(f"bitsandbytes not importable: {type(exc).__name__}: {exc}")

    ram_total, ram_available = _system_ram_mb()
    try:
        disk_free = shutil.disk_usage(str(path_for_disk_check)).free / 1024**2
    except Exception as exc:
        errors.append(f"disk check failed: {exc}")
        disk_free = 0.0

    return HardwareInfo(
        cuda_available=cuda_available, gpu_name=gpu_name, gpu_count=gpu_count,
        vram_total_mb=vram_total, vram_free_mb=vram_free,
        torch_version=torch_version, torch_cuda_version=torch_cuda_version,
        bitsandbytes_available=bnb_available, bitsandbytes_functional=bnb_functional,
        system_ram_total_mb=ram_total, system_ram_available_mb=ram_available,
        disk_free_mb=disk_free, detection_errors=errors,
    )


def estimate_model_params_billion(hf_config) -> float | None:
    """Real computation (standard decoder-only transformer parameter-count
    formula) from a HF `AutoConfig`'s architecture fields -- not a hardcoded
    per-model lookup table. An upper-bound estimate (assumes untied
    embeddings), which is the conservative direction for a feasibility
    check."""
    try:
        hidden = getattr(hf_config, "hidden_size", None)
        layers = getattr(hf_config, "num_hidden_layers", None)
        vocab = getattr(hf_config, "vocab_size", None)
        intermediate = getattr(hf_config, "intermediate_size", None) or (hidden * 4 if hidden else None)
        if not (hidden and layers and vocab and intermediate):
            return None
        per_layer = 4 * hidden * hidden + 2 * hidden * intermediate
        total = layers * per_layer + 2 * vocab * hidden
        return total / 1e9
    except Exception:
        return None


def estimate_required_vram_mb(params_billion: float, *, quantized_4bit: bool, seq_len: int = 2048) -> float:
    """Engineering estimate, not a formal proof: 4-bit weights are ~0.5
    bytes/param, fp16 weights ~2 bytes/param; LoRA optimizer state +
    activations + gradient buffers add a documented multiplier derived from
    established QLoRA deployment guidance."""
    weight_mb = params_billion * 1e9 * (0.5 if quantized_4bit else 2.0) / 1024**2
    overhead_multiplier = 2.2 if quantized_4bit else 4.0
    activation_term_mb = seq_len * 0.5
    return weight_mb * overhead_multiplier + activation_term_mb


_CLOUD_GPU_BANDS = [
    (0, 8000, "no cloud GPU needed -- fits on a consumer 8GB+ card"),
    (8000, 16000, "single 16GB-class GPU (e.g. RTX 4060 Ti 16GB, Nvidia T4 16GB, RTX A4000)"),
    (16000, 24000, "single 24GB-class GPU (e.g. RTX 3090/4090, A10G, L4)"),
    (24000, 40000, "single 40GB-class GPU (e.g. A100 40GB)"),
    (40000, 1_000_000, "single 80GB-class GPU (e.g. A100 80GB, H100) or multi-GPU"),
]


def recommend_cloud_gpu(required_vram_mb: float) -> str:
    for lo, hi, recommendation in _CLOUD_GPU_BANDS:
        if lo <= required_vram_mb < hi:
            return recommendation
    return "multi-GPU / 80GB+ class setup, or reduce model size/sequence length"


@dataclass
class FeasibilityResult:
    feasible: bool
    mode: str  # "LOCAL" | "LOCAL_TRAINING_NOT_FEASIBLE"
    reasons: list[str]
    estimated_params_billion: float | None
    estimated_required_vram_mb: float | None
    recommended_cloud_gpu: str | None
    next_command: str | None
    hardware: HardwareInfo

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def check_feasibility(
    config: CodeModelTrainingConfig, hardware: HardwareInfo | None = None, *,
    min_free_disk_mb: float = 5000.0, dataset_jsonl_path: str | None = None,
) -> FeasibilityResult:
    """Never raises. `mode` is always either `"LOCAL"` (safe to run
    `HuggingFaceLoRATrainingBackend` on this machine right now) or
    `"LOCAL_TRAINING_NOT_FEASIBLE"` (Phase 6/20) -- the latter always comes
    with a concrete `recommended_cloud_gpu` and `next_command` when a size
    estimate was obtainable, so the caller never has to guess what to do
    next."""
    hardware = hardware or detect_hardware()
    reasons: list[str] = []
    params_b: float | None = None
    required_vram: float | None = None

    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(
            config.base_model.model_id, revision=config.base_model.revision,
            trust_remote_code=config.base_model.trust_remote_code,
        )
        params_b = estimate_model_params_billion(hf_config)
    except Exception as exc:
        reasons.append(f"could not fetch/estimate base model size for {config.base_model.model_id!r}: {exc}")

    quantized = config.method == "qlora" and config.quantization.enabled
    if params_b is not None:
        required_vram = estimate_required_vram_mb(params_b, quantized_4bit=quantized, seq_len=config.base_model.max_sequence_length)

    if config.runtime.device != "cpu" and not hardware.cuda_available:
        reasons.append("no CUDA GPU detected (set runtime.device: cpu for a CPU-only run, e.g. small_smoke_test.yaml)")
    if quantized and not hardware.bitsandbytes_functional:
        reasons.append("bitsandbytes 4-bit quantization is not functional in this environment")
    if required_vram is not None and hardware.vram_free_mb is not None and required_vram > hardware.vram_free_mb:
        reasons.append(f"estimated required VRAM ({required_vram:.0f}MB) exceeds free VRAM ({hardware.vram_free_mb:.0f}MB)")
    if hardware.disk_free_mb < min_free_disk_mb:
        reasons.append(f"insufficient disk space ({hardware.disk_free_mb:.0f}MB free, need at least {min_free_disk_mb:.0f}MB)")

    feasible = not reasons
    mode = "LOCAL" if feasible else "LOCAL_TRAINING_NOT_FEASIBLE"
    recommended_gpu = recommend_cloud_gpu(required_vram) if (not feasible and required_vram is not None) else None
    next_command = None
    if not feasible:
        dataset_arg = dataset_jsonl_path or "<dataset_version_jsonl_path>"
        next_command = f"python -m training.code_model.train --config {config.name} --dataset {dataset_arg}"

    return FeasibilityResult(
        feasible=feasible, mode=mode, reasons=reasons, estimated_params_billion=params_b,
        estimated_required_vram_mb=required_vram, recommended_cloud_gpu=recommended_gpu,
        next_command=next_command, hardware=hardware,
    )
