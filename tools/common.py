import torch


def is_ascend_npu():
    """Check if current environment is Ascend NPU"""
    try:
        # Try to import torch_npu
        import torch_npu
        # Check if NPU device is available
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        # torch_npu not installed or NPU not supported
        return False
