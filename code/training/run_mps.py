"""
Launch habitat-baselines forcing PyTorch's MPS (M-series GPU) backend.
Mac-safe: all setup happens inside main() and behind __name__ guard.
"""
import os
import sys


def patch_torch_to_mps():
    """Redirect any torch.device('cuda:*') to MPS instead."""
    import torch

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available.")
        sys.exit(1)

    DEVICE = torch.device("mps")
    print(f"[run_mps] Forcing torch device to: {DEVICE}", flush=True)

    _orig_device = torch.device

    def _patched_device(*args, **kwargs):
        d = _orig_device(*args, **kwargs)
        if d.type == "cuda":
            return DEVICE
        return d

    torch.device = _patched_device


def main():
    # Allow MPS to silently fall back to CPU for unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    patch_torch_to_mps()

    # Hand off to habitat-baselines
    from habitat_baselines.run import main as hb_main
    sys.argv = ["habitat_baselines.run"] + sys.argv[1:]
    hb_main()


if __name__ == "__main__":
    main()
