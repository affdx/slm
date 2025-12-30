"""Device detection utilities for PyTorch.

This module provides utilities for detecting and selecting the best available
compute device (MPS > CUDA > CPU) for PyTorch operations.
"""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Get the best available device for PyTorch operations.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

    Args:
        preferred: Optional preferred device ('mps', 'cuda', 'cpu').
                   If specified and available, this device will be used.

    Returns:
        torch.device: The selected compute device.

    Examples:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> tensor = tensor.to(device)
    """
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "cpu":
            return torch.device("cpu")

    # Auto-detect best available device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """Get detailed information about available compute devices.

    Returns:
        dict: Information about available devices including:
            - selected: The device that would be selected by get_device()
            - mps_available: Whether MPS is available
            - cuda_available: Whether CUDA is available
            - cuda_device_count: Number of CUDA devices (if available)
            - cuda_device_name: Name of CUDA device (if available)
    """
    info = {
        "selected": str(get_device()),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def move_to_device(
    data: torch.Tensor | dict | list | tuple,
    device: torch.device,
) -> torch.Tensor | dict | list | tuple:
    """Recursively move tensors to the specified device.

    Args:
        data: A tensor, dict, list, or tuple containing tensors.
        device: The target device.

    Returns:
        The data structure with all tensors moved to the device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data


if __name__ == "__main__":
    # Print device information when run directly
    info = get_device_info()
    print("Device Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
