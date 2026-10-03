import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _single_gpu_block():
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    block = next(node for node in ast.walk(module) if isinstance(node, ast.If) and "os.name" in ast.unparse(node.test)
                 and "CUDA_VISIBLE_DEVICES" in ast.unparse(node))
    return compile(ast.Module(body=[block], type_ignores=[]), filename=str(main_path), mode="exec")


def _forced_devices(os_name, torch_version, cuda_device=None, environ=None):
    environ = {} if environ is None else dict(environ)
    namespace = {
        "os": SimpleNamespace(name=os_name, environ=environ),
        "args": SimpleNamespace(cuda_device=cuda_device, default_device=None),
        "cuda_malloc": SimpleNamespace(get_torch_version_noimport=lambda: torch_version),
        "logging": SimpleNamespace(warning=lambda msg: None),
    }
    exec(_single_gpu_block(), namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return environ.get("CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize("version", ["2.13.0+rocm10.0.0", "2.9.1+rocmsdk20251116"])
def test_rocm_on_windows_keeps_all_gpus(version):
    # HIP on Windows falls back to CUDA_VISIBLE_DEVICES, so forcing "0" would hide the other AMD GPUs
    assert _forced_devices("nt", version) is None


def test_cuda_on_windows_still_forces_single_gpu():
    assert _forced_devices("nt", "2.10.0+cu130") == "0"


def test_explicit_device_selection_is_left_alone():
    assert _forced_devices("nt", "2.10.0+cu130", cuda_device="1") is None
    assert _forced_devices("nt", "2.10.0+cu130", environ={"CUDA_VISIBLE_DEVICES": "1"}) == "1"
    assert _forced_devices("posix", "2.10.0+cu130") is None
