from datetime import datetime
import os

_run_timestamp = None

def get_run_timestamp() -> str:
    global _run_timestamp
    if _run_timestamp is None:
        _run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return _run_timestamp

def get_output_dir(*parts: str, create: bool = True) -> str:
    """返回带有当前运行时间戳的目录路径，并在需要时创建该目录"""
    path = os.path.join(*parts, get_run_timestamp())
    if create:
        os.makedirs(path, exist_ok=True)
    return path