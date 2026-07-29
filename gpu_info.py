import ctypes
import platform
import re
import threading
import time
from ctypes import wintypes
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Optional


MIB = 1024 * 1024
NVML_LOCK = threading.Lock()
WDDM_COUNTER_PATH = r"\GPU Engine(*)\Utilization Percentage"
WDDM_ENGINE_PATTERN = re.compile(
    r"GPU Engine\([^)]*luid_(?P<luid>0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)"
    r"_phys_(?P<physical>\d+)_eng_(?P<engine>\d+)_engtype_(?P<engine_type>[^)]+)\)",
    re.IGNORECASE,
)


def _is_available(value: Any) -> bool:
    return value is not None and str(value).strip().upper() != "N/A"


def _to_int(value: Any) -> Optional[int]:
    if not _is_available(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if not _is_available(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_string(value: Any) -> Optional[str]:
    if not _is_available(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _bytes_to_mib(value: Any) -> Optional[int]:
    numeric = _to_float(value)
    return round(numeric / MIB) if numeric is not None else None


def _milliwatts_to_watts(value: Any) -> Optional[float]:
    numeric = _to_float(value)
    return round(numeric / 1000, 3) if numeric is not None else None


def _status_string(value: Any) -> Optional[str]:
    if not _is_available(value):
        return None
    if isinstance(value, bool) or value == 0 or value == 1:
        return "Yes" if bool(value) else "No"
    return _to_string(value)


def _safe_call(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return call(*args, **kwargs)
    except Exception:
        return None


def _sequence_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return None


def _attribute_value(value: Any, name: str) -> Any:
    return getattr(value, name, None) if value is not None else None


def _cuda_version_string(value: Any) -> Optional[str]:
    encoded = _to_int(value)
    if encoded is None:
        return None
    return f"{encoded // 1000}.{(encoded % 1000) // 10}"


def _package_version(package_name: str) -> Optional[str]:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _gpu_from_pynvml(pynvml: Any, index: int, driver_version: Optional[str], cuda_version: Optional[str]) -> dict[str, Any]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    memory_info = _safe_call(
        pynvml.nvmlDeviceGetMemoryInfo,
        handle,
        version=getattr(pynvml, "nvmlMemory_v2", None),
    )
    if memory_info is None:
        memory_info = _safe_call(pynvml.nvmlDeviceGetMemoryInfo, handle)

    utilization = _safe_call(pynvml.nvmlDeviceGetUtilizationRates, handle)
    fbc_stats = _safe_call(pynvml.nvmlDeviceGetFBCStats, handle)
    encoder_utilization = _safe_call(pynvml.nvmlDeviceGetEncoderUtilization, handle)
    decoder_utilization = _safe_call(pynvml.nvmlDeviceGetDecoderUtilization, handle)
    encoder_stats = _safe_call(pynvml.nvmlDeviceGetEncoderStats, handle)
    retired_single_bit = _safe_call(
        pynvml.nvmlDeviceGetRetiredPages,
        handle,
        pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
    )
    retired_double_bit = _safe_call(
        pynvml.nvmlDeviceGetRetiredPages,
        handle,
        pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
    )
    retired_pending = _safe_call(pynvml.nvmlDeviceGetRetiredPagesPendingStatus, handle)
    remapped_rows = _safe_call(pynvml.nvmlDeviceGetRemappedRows, handle)

    return {
        "index": index,
        "name": _to_string(_safe_call(pynvml.nvmlDeviceGetName, handle)),
        "uuid": _to_string(_safe_call(pynvml.nvmlDeviceGetUUID, handle)),
        "fan_speed": _to_int(_safe_call(pynvml.nvmlDeviceGetFanSpeed, handle)),
        "fbc_stats_session_count": _to_int(_attribute_value(fbc_stats, "sessionsCount")),
        "fbc_stats_average_fps": _to_int(_attribute_value(fbc_stats, "averageFPS")),
        "fbc_stats_average_latency": _to_int(_attribute_value(fbc_stats, "averageLatency")),
        "memory_free": _bytes_to_mib(_attribute_value(memory_info, "free")),
        "memory_used": _bytes_to_mib(_attribute_value(memory_info, "used")),
        "memory_total": _bytes_to_mib(_attribute_value(memory_info, "total")),
        "memory_reserved": _bytes_to_mib(_attribute_value(memory_info, "reserved")),
        "retired_pages_multiple_single_bit": (
            len(retired_single_bit) if retired_single_bit is not None else None
        ),
        "retired_pages_double_bit": len(retired_double_bit) if retired_double_bit is not None else None,
        "retired_pages_blacklist": None,
        "retired_pages_pending": _status_string(retired_pending),
        "remapped_rows_correctable": _to_int(_sequence_value(remapped_rows, 0)),
        "remapped_rows_uncorrectable": _to_int(_sequence_value(remapped_rows, 1)),
        "remapped_rows_pending": _status_string(_sequence_value(remapped_rows, 2)),
        "remapped_rows_failure": _status_string(_sequence_value(remapped_rows, 3)),
        "power_draw": _milliwatts_to_watts(
            _safe_call(pynvml.nvmlDeviceGetPowerUsage, handle)
        ),
        "power_limit": _milliwatts_to_watts(
            _safe_call(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        ),
        "temperature_gpu": _to_int(
            _safe_call(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
        ),
        "utilization_gpu": _to_int(_attribute_value(utilization, "gpu")),
        "utilization_memory": _to_int(_attribute_value(utilization, "memory")),
        "utilization_encoder": _to_int(_sequence_value(encoder_utilization, 0)),
        "utilization_decoder": _to_int(_sequence_value(decoder_utilization, 0)),
        "pcie_link_gen_current": _to_int(
            _safe_call(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle)
        ),
        "pcie_link_width_current": _to_int(
            _safe_call(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle)
        ),
        "encoder_stats_session_count": _to_int(_sequence_value(encoder_stats, 0)),
        "encoder_stats_average_fps": _to_int(_sequence_value(encoder_stats, 1)),
        "encoder_stats_average_latency": _to_int(_sequence_value(encoder_stats, 2)),
        "clocks_current_graphics": _to_int(
            _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS)
        ),
        "clocks_current_sm": _to_int(
            _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM)
        ),
        "clocks_current_memory": _to_int(
            _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM)
        ),
        "clocks_current_video": _to_int(
            _safe_call(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_VIDEO)
        ),
        "driver_version": driver_version,
        "cuda_version": cuda_version,
    }


def _collect_pynvml_info() -> dict[str, Any]:
    try:
        import pynvml
    except Exception as exc:
        return {
            "available": False,
            "provider": "pynvml",
            "binding_version": None,
            "driver_version": None,
            "cuda_version": None,
            "gpus": [],
            "error": str(exc),
        }

    with NVML_LOCK:
        try:
            pynvml.nvmlInit()
            driver_version = _to_string(pynvml.nvmlSystemGetDriverVersion())
            cuda_version_call = getattr(
                pynvml,
                "nvmlSystemGetCudaDriverVersion_v2",
                pynvml.nvmlSystemGetCudaDriverVersion,
            )
            cuda_version = _cuda_version_string(_safe_call(cuda_version_call))
            gpus = [
                _gpu_from_pynvml(pynvml, index, driver_version, cuda_version)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception as exc:
            return {
                "available": False,
                "provider": "pynvml",
                "binding_version": _package_version("nvidia-ml-py"),
                "driver_version": None,
                "cuda_version": None,
                "gpus": [],
                "error": str(exc),
            }
        finally:
            _safe_call(pynvml.nvmlShutdown)

    return {
        "available": bool(gpus),
        "provider": "pynvml",
        "binding_version": _package_version("nvidia-ml-py"),
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "gpus": gpus,
        "error": None if gpus else "no GPU returned by NVML",
    }


def get_windows_wddm_gpu_info(sample_interval_seconds: float = 0.2) -> dict[str, Any]:
    """Read the Windows Task Manager GPU Engine counters through PDH."""
    if platform.system() != "Windows":
        return {
            "available": False,
            "provider": "windows-pdh",
            "utilization_gpu": None,
            "gpus": [],
            "error": "WDDM counters are only available on Windows",
        }

    pdh = ctypes.WinDLL("pdh.dll")
    pdh_more_data = 0x800007D2
    pdh_format_double = 0x00000200

    class PdhFormattedCounterValue(ctypes.Structure):
        _fields_ = [
            ("status", wintypes.DWORD),
            ("double_value", ctypes.c_double),
        ]

    query_handle = wintypes.HANDLE()
    counter_type = wintypes.DWORD()
    buffer_size = wintypes.DWORD(0)

    pdh.PdhOpenQueryW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_size_t,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    pdh.PdhOpenQueryW.restype = wintypes.DWORD
    pdh.PdhExpandWildCardPathW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    pdh.PdhExpandWildCardPathW.restype = wintypes.DWORD
    pdh.PdhAddEnglishCounterW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCWSTR,
        ctypes.c_size_t,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
    pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
    pdh.PdhCollectQueryData.restype = wintypes.DWORD
    pdh.PdhGetFormattedCounterValue.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(PdhFormattedCounterValue),
    ]
    pdh.PdhGetFormattedCounterValue.restype = wintypes.DWORD
    pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
    pdh.PdhCloseQuery.restype = wintypes.DWORD

    status = pdh.PdhExpandWildCardPathW(
        None,
        WDDM_COUNTER_PATH,
        None,
        ctypes.byref(buffer_size),
        0,
    )
    if status != pdh_more_data:
        return {
            "available": False,
            "provider": "windows-pdh",
            "utilization_gpu": None,
            "gpus": [],
            "error": f"cannot enumerate WDDM counters: 0x{status:08X}",
        }

    buffer = ctypes.create_unicode_buffer(buffer_size.value)
    status = pdh.PdhExpandWildCardPathW(
        None,
        WDDM_COUNTER_PATH,
        buffer,
        ctypes.byref(buffer_size),
        0,
    )
    if status != 0:
        return {
            "available": False,
            "provider": "windows-pdh",
            "utilization_gpu": None,
            "gpus": [],
            "error": f"cannot expand WDDM counters: 0x{status:08X}",
        }

    counter_paths = [
        path
        for path in buffer[: buffer_size.value].split("\0")
        if path
    ]
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query_handle)) != 0:
        return {
            "available": False,
            "provider": "windows-pdh",
            "utilization_gpu": None,
            "gpus": [],
            "error": "cannot open WDDM performance query",
        }

    counters: list[tuple[str, wintypes.HANDLE]] = []
    try:
        for path in counter_paths:
            counter_handle = wintypes.HANDLE()
            if (
                pdh.PdhAddEnglishCounterW(
                    query_handle,
                    path,
                    0,
                    ctypes.byref(counter_handle),
                )
                == 0
            ):
                counters.append((path, counter_handle))

        if not counters:
            raise RuntimeError("no WDDM GPU Engine counters found")
        if pdh.PdhCollectQueryData(query_handle) != 0:
            raise RuntimeError("cannot collect the first WDDM sample")
        time.sleep(max(0.05, min(sample_interval_seconds, 1.0)))
        if pdh.PdhCollectQueryData(query_handle) != 0:
            raise RuntimeError("cannot collect the second WDDM sample")

        engine_totals: dict[tuple[str, int, int, str], float] = {}
        for path, counter_handle in counters:
            value = PdhFormattedCounterValue()
            if (
                pdh.PdhGetFormattedCounterValue(
                    counter_handle,
                    pdh_format_double,
                    ctypes.byref(counter_type),
                    ctypes.byref(value),
                )
                != 0
            ):
                continue
            match = WDDM_ENGINE_PATTERN.search(path)
            if match is None:
                continue
            key = (
                match.group("luid").lower(),
                int(match.group("physical")),
                int(match.group("engine")),
                match.group("engine_type"),
            )
            engine_totals[key] = engine_totals.get(key, 0.0) + max(
                0.0,
                value.double_value,
            )

        adapter_engines: dict[tuple[str, int], list[tuple[float, int, str]]] = {}
        for (luid, physical, engine, engine_type), utilization in engine_totals.items():
            adapter_engines.setdefault((luid, physical), []).append(
                (min(utilization, 100.0), engine, engine_type)
            )

        gpus = []
        for (luid, physical), engines in adapter_engines.items():
            utilization, engine, engine_type = max(engines, default=(0.0, 0, "Unknown"))
            gpus.append(
                {
                    "luid": luid,
                    "physical_index": physical,
                    "utilization_gpu": round(utilization, 2),
                    "busiest_engine_id": engine,
                    "busiest_engine_type": engine_type,
                }
            )
        gpus.sort(key=lambda gpu: (gpu["luid"], gpu["physical_index"]))
        utilization_gpu = max(
            (gpu["utilization_gpu"] for gpu in gpus),
            default=0.0,
        )
    except Exception as exc:
        return {
            "available": False,
            "provider": "windows-pdh",
            "utilization_gpu": None,
            "gpus": [],
            "error": str(exc),
        }
    finally:
        pdh.PdhCloseQuery(query_handle)

    return {
        "available": bool(gpus),
        "provider": "windows-pdh",
        "utilization_gpu": utilization_gpu,
        "gpus": gpus,
        "error": None if gpus else "no WDDM GPU Engine data returned",
    }


def get_pynvml_gpu_info() -> dict[str, Any]:
    """Return NVML metrics plus Windows Task Manager-compatible utilization."""
    nvml_info = _collect_pynvml_info()
    wddm_info = get_windows_wddm_gpu_info()
    nvml_max = max(
        (
            gpu["utilization_gpu"]
            for gpu in nvml_info["gpus"]
            if gpu["utilization_gpu"] is not None
        ),
        default=None,
    )
    wddm_max = wddm_info["utilization_gpu"]
    use_wddm = platform.system() == "Windows" and wddm_max is not None
    scheduling_utilization = wddm_max if use_wddm else nvml_max
    scheduling_source = "wddm" if use_wddm else "nvml"
    if scheduling_utilization is None:
        scheduling_source = None

    if len(nvml_info["gpus"]) == 1 and wddm_max is not None:
        gpu = nvml_info["gpus"][0]
        gpu["utilization_wddm"] = wddm_max
        gpu["scheduling_utilization"] = (
            gpu["utilization_wddm"] if use_wddm else gpu["utilization_gpu"]
        )
    else:
        for gpu in nvml_info["gpus"]:
            gpu["utilization_wddm"] = None
            gpu["scheduling_utilization"] = (
                wddm_max if use_wddm else gpu["utilization_gpu"]
            )

    return {
        **nvml_info,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "gpu_count": len(nvml_info["gpus"]),
        "nvml_utilization_gpu": nvml_max,
        "wddm_utilization_gpu": wddm_max,
        "scheduling_utilization": scheduling_utilization,
        "scheduling_source": scheduling_source,
        "wddm": wddm_info,
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
                    "memory_total_mb": round(props.total_memory / MIB),
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

    The timeout argument is retained for compatibility. NVML and PDH are queried
    directly without starting a subprocess.
    """
    del timeout_seconds
    pynvml_info = get_pynvml_gpu_info()
    torch_cuda = _get_torch_cuda_info()
    primary = pynvml_info if pynvml_info["available"] else torch_cuda

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
        "pynvml": pynvml_info,
        "torch_cuda": torch_cuda,
    }
