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
from typing import Any


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



def load_tensor_by_name(
    *,
    tensor_name: str,
    model_dir: str | Path,
    manifest_path: str | Path,
    device: str = "cpu",
):
    """根据 tensor_name 从 safetensors 分片中读取一个 torch.Tensor。"""

    # 在函数内部导入 torch，这样 --help 或静态检查不需要提前加载 torch。
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("加载权重需要安装 PyTorch，请先执行：pip install torch") from exc

    # 读取 manifest 中该 tensor 对应的位置信息。
    entry = read_manifest_entry(manifest_path=Path(manifest_path), tensor_name=tensor_name)

    # manifest 里只保存分片文件名，因此这里用 model_dir 拼出真实分片路径。
    shard_path = Path(model_dir) / entry["file"]

    # 检查分片文件是否存在，避免后面 open 报一个不够直观的错误。
    if not shard_path.is_file():
        raise FileNotFoundError(f"Safetensors shard does not exist: {shard_path}")

    dtype_name = entry["dtype"]

    # 把 safetensors dtype 转成 torch dtype
    torch_dtype = to_torch_dtype(torch, dtype_name)

    shape = entry["shape"]

    offset = entry["offset"]

    nbytes = entry["nbytes"]

    validate_tensor_size(dtype_name=dtype_name, shape=shape, nbytes=nbytes, tensor_name=tensor_name)

    with shard_path.open("rb") as handle:
        # 移动到该 tensor 的文件内绝对偏移。
        handle.seek(offset)

        # 精确读取该 tensor 占用的字节数。
        raw_bytes = handle.read(nbytes)

    if len(raw_bytes) != nbytes:
        raise ValueError(
            f"Read {len(raw_bytes)} bytes for {tensor_name}, but manifest expects {nbytes} bytes."
        )

    # torch.frombuffer 会把 bytes 直接解释成一维 tensor；clone() 让 tensor 拥有独立可写内存。
    tensor = torch.frombuffer(raw_bytes, dtype=torch_dtype)

    # 按 manifest 记录的 shape 还原 tensor 形状。
    tensor = tensor.reshape(shape)

    # 如果用户指定 cuda 等设备，这里再把 tensor 移动过去。
    return tensor.to(device)


def read_manifest_entry(*, manifest_path: Path, tensor_name: str) -> dict[str, Any]:
    """从 manifest 中取出某个 tensor 的条目。"""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}. "
        )

    # 按 UTF-8 读取 JSON，顶层应该是 {tensor_name: entry}。
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    # 如果 tensor_name 不存在，说明名字拼错或 manifest 不是当前模型生成的。
    if tensor_name not in manifest:
        raise KeyError(f"Tensor name not found in manifest: {tensor_name}")

    # 返回该 tensor 的 file、offset、nbytes、dtype、shape 信息。
    return manifest[tensor_name]


def to_torch_dtype(torch_module: Any, dtype_name: str):
    """把 safetensors dtype 名转换成 torch dtype。"""

    # 如果遇到未登记的 dtype，直接报错，避免悄悄按错误类型解释二进制数据。
    if dtype_name not in TORCH_DTYPE_NAMES:
        supported = ", ".join(sorted(TORCH_DTYPE_NAMES))
        raise ValueError(f"Unsupported dtype {dtype_name}. Supported dtypes: {supported}")

    # getattr(torch, "bfloat16") -> torch.bfloat16。
    return getattr(torch_module, TORCH_DTYPE_NAMES[dtype_name])


def validate_tensor_size(*, dtype_name: str, shape: list[int], nbytes: int, tensor_name: str) -> None:
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

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 这个脚本会真正把某个权重读成 torch.Tensor，因此运行时需要安装 torch。
    parser = argparse.ArgumentParser(
        description="根据 tensor_name 从本地 safetensors 分片中精确读取一个权重 tensor。"
    )

    # tensor 名必须显式传入，例如 model.layers.0.mlp.experts.0.gate_proj.weight。
    parser.add_argument(
        "tensor_name",
        help="要读取的完整 tensor 名称。",
    )

    # 模型目录中需要包含 weight_manifest.json 和对应的 safetensors 分片。
    parser.add_argument(
        "--model-dir",
        default="models/Qwen1.5-Moe-A2.7B",
        help="模型目录，默认读取 models/Qwen1.5-Moe-A2.7B。",
    )

    # manifest 路径可选；不传时默认使用模型目录下的 weight_manifest.json。
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest JSON 路径，默认使用 <model-dir>/weight_manifest.json。",
    )

    # 默认先读到 CPU；后续如果要直接放 GPU，可以传 --device cuda。
    parser.add_argument(
        "--device",
        default="cpu",
        help="加载后的目标设备，例如 cpu、cuda、cuda:0。",
    )

    # 默认只打印 tensor 元信息，不打印值，避免误把大权重刷满终端。
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="打印 tensor 内容。大 tensor 不建议开启。",
    )

    return parser.parse_args()


def main() -> None:
    """命令行入口：读取一个 tensor，并打印基本信息。"""

    # 解析用户命令行输入。
    args = parse_args()

    # 统一把模型目录解析成绝对路径，后续拼接分片文件更稳。
    model_dir = Path(args.model_dir).resolve()

    # 如果用户传了 manifest 路径，就使用用户路径；否则默认使用模型目录下的 weight_manifest.json。
    manifest_path = Path(args.manifest).resolve() if args.manifest else model_dir / "weight_manifest.json"

    # 通过 tensor_name 从 manifest 定位、读取并还原成 torch.Tensor。
    tensor = load_tensor_by_name(
        tensor_name=args.tensor_name,
        model_dir=model_dir,
        manifest_path=manifest_path,
        device=args.device,
    )

    # 打印核心信息，方便确认 dtype、shape、device 是否符合预期。
    print(f"name: {args.tensor_name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(f"dtype: {tensor.dtype}")
    print(f"device: {tensor.device}")

    # 如果用户明确要求打印值，才输出 tensor 内容。
    if args.print_values:
        print(tensor)


# 只有直接执行该脚本时才运行 main()；被其他代码 import 时只提供 load_tensor_by_name 函数。
if __name__ == "__main__":
    main()
