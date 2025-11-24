import torch
import sys
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class HardwareConfig:
    device: str
    dtype: torch.dtype
    use_scaler: bool
    compile_mode: Optional[str]


def detect_hardware_config() -> HardwareConfig:
    if not torch.cuda.is_available():
        return HardwareConfig(
            device="cpu",
            dtype=torch.float32,
            use_scaler=False,
            compile_mode=None,
            memory_format=torch.contiguous_format,
        )

    device_name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    print(
        f"[Hardware] Detected GPU: {device_name} (Compute Capability {cap_major}.{cap_minor})"
    )

    dtype = torch.float16
    use_scaler = True
    compile_mode = None

    # --- 1. Precyzja (BF16 vs FP16) ---
    # Ampere (8.0) i nowsze (np. RTX 30xx, A100, H100, Blackwell) wspierają BF16 sprzętowo
    if cap_major >= 8 and torch.cuda.is_bf16_supported():
        print("[Hardware] Ampere+ detected. Using BFloat16 (No Scaler).")
        dtype = torch.bfloat16
        use_scaler = False  # BF16 nie wymaga scalera
    else:
        # Pascal (6.0, 6.1), Volta (7.0), Turing (7.5)
        print("[Hardware] Pre-Ampere detected. Using Float16 + GradScaler.")
        dtype = torch.float16
        use_scaler = True

    # --- 2. Kompilacja (torch.compile) ---
    # Wymaga Volta (7.0)+. Pascal (GTX 10xx, P100) nie jest wspierany przez Triton.
    if sys.platform.startswith("win"):
        print("[Hardware] Windows detected. Disabling torch.compile.")
        compile_mode = None
    elif cap_major < 7:
        print("[Hardware] Pascal GPU (GTX 10xx/P100). Disabling torch.compile.")
        compile_mode = None
    else:
        # T4 (7.5) lub nowsze
        if cap_major >= 8:
            # RTX 30xx/40xx/50xx są bardzo szybkie, max-autotune wyciśnie max
            compile_mode = "max-autotune"
        else:
            # T4 (Turing) bywa kapryśne z max-autotune, bezpieczniej reduce-overhead
            compile_mode = "reduce-overhead"
        print(f"[Hardware] Enabling torch.compile(mode='{compile_mode}').")

    return HardwareConfig(
        device="cuda",
        dtype=dtype,
        use_scaler=use_scaler,
        compile_mode=compile_mode,
    )


def safe_compile_model(model: torch.nn.Module, config: HardwareConfig) -> torch.nn.Module:
    """Wrapper aplikujący kompilację tylko jeśli konfiguracja na to pozwala."""
    if config.compile_mode:
        try:
            return torch.compile(model, mode=config.compile_mode)
        except Exception as e:
            print(f"[Warning] Compilation failed: {e}. Running eager.")
            return model
    return model
