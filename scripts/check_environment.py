"""Import and CPU smoke check; no market data or model training."""

import importlib
import importlib.metadata
import json
import platform
import sys

PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "PyYAML": "yaml",
    "pytest": "pytest",
}


def main():
    versions = {}
    for distribution, module in PACKAGES.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)

    import torch

    tensor = torch.arange(6, dtype=torch.float32, device="cpu").reshape(2, 3)
    assert tensor.device.type == "cpu"
    assert tensor.square().sum().item() == 55.0
    assert torch.version.cuda is None, "Expected a CPU-only PyTorch build"
    print(json.dumps({
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "cpu_tensor": tensor.tolist(),
        "cpu_sum_of_squares": tensor.square().sum().item(),
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
