"""
Launch habitat-baselines with:
  - simulator running in the same process (no multiprocessing) -> avoids macOS Metal issue
  - torch.load patched to weights_only=False -> works with PyTorch 2.6+ checkpoint format
CPU-only PyTorch. habitat-sim still uses M2 GPU for rendering in-process.
"""
import os
import sys


def patch_torch_load_weights_only():
    """Force torch.load to default to weights_only=False (legacy behavior).
    Needed because habitat-baselines saves DictConfig in checkpoints."""
    import torch
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load


def patch_to_threaded_vector_env():
    """Replace VectorEnv with ThreadedVectorEnv to avoid macOS Metal-in-fork issue."""
    import habitat
    from habitat.core.vector_env import VectorEnv, ThreadedVectorEnv  # noqa: F401

    import habitat.core.vector_env as _ve_mod
    _ve_mod.VectorEnv = ThreadedVectorEnv

    import habitat_baselines.common.habitat_env_factory as _ef_mod
    if hasattr(_ef_mod, "VectorEnv"):
        _ef_mod.VectorEnv = ThreadedVectorEnv
    if hasattr(_ef_mod, "habitat"):
        try:
            _ef_mod.habitat.VectorEnv = ThreadedVectorEnv
        except Exception:
            pass


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HABITAT_ENV_DEBUG", "1")

    patch_torch_load_weights_only()
    patch_to_threaded_vector_env()

    from habitat_baselines.run import main as hb_main
    sys.argv = ["habitat_baselines.run"] + sys.argv[1:]
    hb_main()


if __name__ == "__main__":
    main()
