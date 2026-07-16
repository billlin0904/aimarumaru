import platform
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional


NVIDIA_SMI_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
]


def _to_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _run_nvidia_smi(timeout_seconds: int) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "available": False,
            "provider": "nvidia-smi",
            "gpus": [],
            "error": "nvidia-smi not found",
        }

    command = [
        nvidia_smi,
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=True,
        )
    except Exception as exc:
        return {
            "available": False,
            "provider": "nvidia-smi",
            "gpus": [],
            "error": str(exc),
        }

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        values.extend([""] * (len(NVIDIA_SMI_FIELDS) - len(values)))
        row = dict(zip(NVIDIA_SMI_FIELDS, values))
        gpus.append(
            {
                "index": _to_int(row["index"]),
                "name": row["name"] or None,
                "uuid": row["uuid"] or None,
                "driver_version": row["driver_version"] or None,
                "memory_total_mb": _to_int(row["memory.total"]),
                "memory_used_mb": _to_int(row["memory.used"]),
                "memory_free_mb": _to_int(row["memory.free"]),
                "gpu_utilization_percent": _to_int(row["utilization.gpu"]),
                "memory_utilization_percent": _to_int(row["utilization.memory"]),
                "temperature_c": _to_int(row["temperature.gpu"]),
                "power_draw_w": _to_float(row["power.draw"]),
                "power_limit_w": _to_float(row["power.limit"]),
            }
        )

    return {
        "available": bool(gpus),
        "provider": "nvidia-smi",
        "gpus": gpus,
        "error": None if gpus else "no gpu returned by nvidia-smi",
    }


def _get_torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "provider": "torch",
            "cuda_available": False,
            "gpus": [],
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_mb": round(props.total_memory / 1024 / 1024),
                }
            )

    return {
        "available": cuda_available,
        "provider": "torch",
        "cuda_available": cuda_available,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpus": gpus,
        "error": None,
    }


def get_local_gpu_info(timeout_seconds: int = 3) -> dict[str, Any]:
    """
    Return local GPU information for scheduling or diagnostics.

    NVIDIA cards use nvidia-smi when available because it exposes live memory,
    utilization, temperature, and power data. Torch CUDA is used as a fallback
    so the function still reports CUDA capability when nvidia-smi is missing.
    """
    nvidia = _run_nvidia_smi(timeout_seconds)
    torch_cuda = _get_torch_cuda_info()
    primary = nvidia if nvidia["available"] else torch_cuda

    return {
        "available": bool(primary["available"]),
        "provider": primary["provider"],
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "gpu_count": len(primary["gpus"]),
        "gpus": primary["gpus"],
        "nvidia_smi": nvidia,
        "torch_cuda": torch_cuda,
    }
