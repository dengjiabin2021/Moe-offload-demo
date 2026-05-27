from __future__ import annotations

# argparse 用来解析命令行参数，例如 tensor 名、模型目录、目标设备等。
import argparse
# json 用来读取 build_safetensors_manifest.py 生成的 weight_manifest.json。
import json
# math.prod 用来计算 shape 中所有维度的乘积，也就是 tensor 元素个数。
from math import prod
# Path 用来安全地拼接模型目录、manifest 路径和分片文件路径。
from pathlib import Path
# Any 用在类型标注中，表示 JSON 里可能有字符串、数字、列表、字典等类型。
from typing import Any, Final, Dict
import torch
import mmap  # 新增：内存映射模块（Python内置，无需安装）

# safetensors 的 dtype 名称到 PyTorch dtype 名称的映射。
# 这里先覆盖 Qwen1.5-MoE-A2.7B 常见的 BF16，同时也保留其他常见类型，方便复用。
TORCH_DTYPE_NAMES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


# 每种 safetensors dtype 对应的单个元素字节数。
# 这个表用于在真正构造 tensor 前检查 manifest 是否和 shape 对得上。
DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


class ManifestTensorLoader:

    """基于 weight_manifest.json，通过 tensor_name 精确读取单个 tensor。"""
    def __init__(self, *, 
                 model_dir: str | Path, 
                 manifest_path: str | Path) -> None:

        self.model_dir = Path(model_dir)
        self.manifest_path = Path(manifest_path)
        self._manifest: Dict[str, Dict[str, Any]] | None = None
        self._load_manifest_json()
        
    def load_tensors_by_names(self, *, tensor_names: list[str], device: str) -> dict[str, torch.Tensor]:
        """根据 tensor_names 从 safetensors 分片中读取一系列 torch.Tensor。"""

        if self._manifest is None:
            self._load_manifest_json()

        # 1. 收集所有需要的条目，并按分片分组
        shard_groups: dict[Path, list[tuple[str, dict]]] = {}
        for name in tensor_names:
            entry = self._read_manifest_entry(tensor_name=name)
            shard_path = self.model_dir / entry["file"]
            shard_groups.setdefault(shard_path, []).append((name, entry))

        result: dict[str, torch.Tensor] = {}

        # 2. 逐分片处理
        for shard_path, items in shard_groups.items():
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard not found: {shard_path}")

            # 按 offset 排序，利于顺序读取
            items.sort(key=lambda x: x[1]["offset"])

            with open(shard_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for name, entry in items:
                        dtype_name = entry["dtype"]
                        torch_dtype = self._to_torch_dtype(dtype_name)
                        offset = entry["offset"]
                        nbytes = entry["nbytes"]
                        element_size = torch.empty((), dtype=torch_dtype).element_size()

                        # 从 mmap 中切片并 clone（clone 后才拥有独立内存）
                        tensor = torch.frombuffer(mm, dtype=torch_dtype, offset=offset,
                                                count=nbytes // element_size).clone()
                        tensor = tensor.reshape(entry["shape"]).to(device)
                        result[name] = tensor

        # 确保所有请求的名称都有返回（可通过检查 result 长度）
        return result

    def load_tensor_by_name(self, *, tensor_name: str, device: str) -> torch.Tensor:
        """根据 tensor_name 从 safetensors 分片中读取一个 torch.Tensor。"""

        if self._manifest is None:
            self._load_manifest_json()

        # 读取 manifest 中该 tensor 对应的位置信息。
        entry = self._read_manifest_entry(tensor_name=tensor_name)

        # manifest 里只保存分片文件名，因此这里用 model_dir 拼出真实分片路径。
        shard_path = Path(self.model_dir) / entry["file"]
        # 检查分片文件是否存在，避免后面 open 报一个不够直观的错误。
        if not shard_path.is_file():
            raise FileNotFoundError(f"Safetensors shard does not exist: {shard_path}")

        dtype_name = entry["dtype"]

        # 把 safetensors dtype 转成 torch dtype
        torch_dtype = self._to_torch_dtype(dtype_name)

        shape = entry["shape"]

        offset = entry["offset"]

        nbytes = entry["nbytes"]

        self._validate_tensor_size(dtype_name=dtype_name, shape=shape, nbytes=nbytes, tensor_name=tensor_name)

        element_size = torch.empty((), dtype=torch_dtype).element_size()

        with open(shard_path, "rb") as f:
            # 内存映射文件（不加载数据到内存，仅建立虚拟地址映射）
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                # 直接从映射内存创建张量：零拷贝，无警告，无冗余内存
                tensor = torch.frombuffer(
                    mm,
                    dtype=torch_dtype,
                    offset=offset,
                    count=nbytes // element_size
                ).clone()  # clone后获得独立内存，退出with块自动释放mmap映射

        # 按 manifest 记录的 shape 还原 tensor 形状。
        tensor = tensor.reshape(shape)

        # 如果用户指定 cuda 等设备，这里再把 tensor 移动过去。
        return tensor.to(device)

    def _load_manifest_json(self) -> None:

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest does not exist: {self.manifest_path}. "
            )

        # 按 UTF-8 读取 JSON，顶层应该是 {tensor_name: entry}。
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self._manifest = json.load(handle)

    def _read_manifest_entry(self, tensor_name: str) -> dict[str, Any]:
        """从 manifest 中取出某个 tensor 的条目。"""

        # 如果 tensor_name 不存在，说明名字拼错或 manifest 不是当前模型生成的。
        if tensor_name not in self._manifest:
            raise KeyError(f"Tensor name not found in manifest: {tensor_name}")

        # 返回该 tensor 的 file、offset、nbytes、dtype、shape 信息。
        return self._manifest[tensor_name]


    def _to_torch_dtype(self, dtype_name: str) -> torch.dtype:
        """把 safetensors dtype 名转换成 torch dtype。"""

        if dtype_name not in TORCH_DTYPE_NAMES:
            supported = ", ".join(sorted(TORCH_DTYPE_NAMES))
            raise ValueError(f"Unsupported dtype {dtype_name}. Supported dtypes: {supported}")

        # getattr(torch, "bfloat16") -> torch.bfloat16。
        return getattr(torch, TORCH_DTYPE_NAMES[dtype_name])


    def _validate_tensor_size(self, *, dtype_name: str, shape: list[int], nbytes: int, tensor_name: str) -> None:
        """校验 manifest 中的 shape、dtype 和 nbytes 是否一致。"""

        # 如果 dtype 未登记，无法计算单元素字节数，直接报错。
        if dtype_name not in DTYPE_NBYTES:
            supported = ", ".join(sorted(DTYPE_NBYTES))
            raise ValueError(f"Unsupported dtype {dtype_name}. Supported dtypes: {supported}")

        # numel 是 tensor 元素个数，例如 [1408, 2048] 的乘积。
        numel = prod(shape)

        # 预期字节数 = 元素个数 * 单元素字节数。
        expected_nbytes = numel * DTYPE_NBYTES[dtype_name]

        # 如果预期字节数和 manifest 中的 nbytes 不一致，说明数据解释方式不可靠。
        if expected_nbytes != nbytes:
            raise ValueError(
                f"Size mismatch for {tensor_name}: shape={shape}, dtype={dtype_name}, "
                f"expected {expected_nbytes} bytes, got {nbytes} bytes."
            )

