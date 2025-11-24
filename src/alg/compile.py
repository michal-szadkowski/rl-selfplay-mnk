import torch
import sys
import platform


def safe_compile(model, mode="reduce-overhead"):
    try:
        version_major = int(torch.__version__.split(".")[0])
    except:
        version_major = 1  # Fallback dla dziwnych stringów wersji

    if version_major < 2:
        print(
            f"[Info] PyTorch {torch.__version__} detected. torch.compile not supported. Skipping."
        )
        return model

    if sys.platform.startswith("win"):
        print(
            "[Info] Windows detected. torch.compile support is experimental. Skipping to ensure stability."
        )
        return model

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            print(f"[Info] GPU Compute Capability is {major}.{minor} (Pascal or older).")
            print(
                "[Info] torch.compile (Triton) requires Capability >= 7.0 (Volta+). Skipping compilation."
            )
            return model

    try:
        if hasattr(torch, "compile"):
            print(f"[Info] Compiling model with torch.compile(mode='{mode}')...")

            compiled_model = torch.compile(model, mode=mode)

            return compiled_model
        else:
            return model

    except Exception as e:
        print(f"[Warning] torch.compile failed with error: {e}")
        print("[Info] Falling back to eager execution.")
        return model
