from __future__ import annotations

# argparse 用来解析命令行参数，例如模型目录和输出文件路径。
import argparse
# json 用来解析 safetensors header，也用来写出最终的 manifest 文件。
import json
# struct 用来按照二进制格式读取 safetensors 文件开头的 8 字节 header 长度。
import struct
# Path 比字符串路径更安全，能跨平台拼接、解析和遍历文件。
from pathlib import Path
# Any 用在类型标注里，表示 JSON 中可能出现字符串、数字、列表、字典等多种类型。
from typing import Any


# safetensors 文件格式规定：
# 文件最开头的 8 个字节是 little-endian uint64，表示后面 JSON header 的字节长度。
# 真正的 tensor 二进制数据紧跟在这段 JSON header 后面。
SAFETENSORS_HEADER_LEN_BYTES = 8


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 这里的脚本只负责生成索引文件，不读取 tensor 内容，所以不需要 torch/safetensors 依赖。
    parser = argparse.ArgumentParser(
        description=(
            "读取本地 safetensors 分片的 header，生成每个权重的文件名、"
            "文件内偏移、字节数、dtype 和 shape。"
        )
    )

    # 模型目录：里面应该包含 model-00001-of-00008.safetensors 这类分片文件。
    parser.add_argument(
        "--model-dir",
        default="models/Qwen1.5-Moe-A2.7B",
        help="包含 .safetensors 分片的模型目录。",
    )

    # 输出路径：不传时默认写到模型目录下，方便和权重文件放在一起管理。
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径，默认写到 <model-dir>/weight_manifest.json。",
    )

    # 返回 argparse.Namespace，后续通过 args.model_dir / args.output 访问。
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析参数、构建 manifest、写入 JSON 文件。"""

    # 读取用户传入的命令行参数。
    args = parse_args()

    # resolve() 会把相对路径转换成绝对路径，便于后续检查目录是否存在。
    model_dir = Path(args.model_dir).resolve()

    # 如果用户指定了 --output，就用用户给的路径；否则写到模型目录下的 weight_manifest.json。
    output_path = Path(args.output).resolve() if args.output else model_dir / "weight_manifest.json"

    # 扫描所有 safetensors 分片，生成 {权重名: 位置信息} 的字典。
    manifest = build_manifest(model_dir)

    # 确保输出文件所在目录存在；如果目录已经存在，不会报错。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 UTF-8 写 JSON，indent=2 让不同字段换行，方便人工检查和 git diff。
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

        # 末尾补一个换行，符合常见文本文件规范。
        handle.write("\n")

    print(f"Wrote {len(manifest)} tensor entries to {output_path}")


def build_manifest(model_dir: Path) -> dict[str, dict[str, Any]]:
    """遍历模型目录下的所有 safetensors 分片，合并成完整权重 manifest。"""

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    # 按文件名排序，保证每次生成 JSON 的顺序稳定，便于对比和调试。
    shard_paths = sorted(model_dir.glob("*.safetensors"))

    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files found in: {model_dir}")


    weights: dict[str, dict[str, Any]] = {}

    # 逐个分片读取 header。注意这里只读 header，不读取真实权重数据。
    for shard_path in shard_paths:
        # 解析单个 safetensors 文件中的所有 tensor 元数据。
        shard_entries = read_safetensors_header(model_dir, shard_path)

        # 把当前分片的条目合并到总 manifest 里。
        for tensor_name, entry in shard_entries.items():
            # 正常的 safetensors 分片中，同一个 tensor 名不应该出现在多个文件里。
            # 如果重复，后续按名字读取会有歧义，所以这里直接报错。
            if tensor_name in weights:
                raise ValueError(f"Duplicate tensor name across shards: {tensor_name}")

            # 保存该 tensor 的磁盘位置信息。
            weights[tensor_name] = entry

    # 顶层只返回权重映射本身，不额外加入 format/model_dir/统计信息，保持文件简洁。
    # sorted() 保证 key 顺序稳定，生成的 JSON 更好读。
    return dict(sorted(weights.items()))


def read_safetensors_header(model_dir: Path, shard_path: Path) -> dict[str, dict[str, Any]]:
    """读取单个 safetensors 文件头，返回其中每个 tensor 的磁盘位置信息。"""

    # safetensors 是二进制文件，所以用 rb 模式读取。
    with shard_path.open("rb") as handle:
        # 先读取文件开头 8 字节，这 8 字节记录 JSON header 的长度。
        header_len_bytes = handle.read(SAFETENSORS_HEADER_LEN_BYTES)

        # 如果连 8 字节都读不到，说明这个文件不可能是合法 safetensors。
        if len(header_len_bytes) != SAFETENSORS_HEADER_LEN_BYTES:
            raise ValueError(f"File is too small to be safetensors: {shard_path}")

        # "<Q" 表示 little-endian unsigned long long，也就是 safetensors 规定的 header 长度格式。
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        # 根据刚刚读出的长度，继续读取 JSON header 内容。
        header_bytes = handle.read(header_len)

        # 如果实际读到的 header 字节数不够，说明文件可能损坏或被截断。
        if len(header_bytes) != header_len:
            raise ValueError(f"Incomplete safetensors header in: {shard_path}")


    header = json.loads(header_bytes.decode("utf-8"))

    # data_offsets 里的 start/end 是相对“数据区起点”的偏移。
    # 数据区起点 = 8 字节 header 长度字段 + JSON header 本身长度。
    # 所以后续真正 open 文件并 seek 时，需要使用 data_start + start。
    data_start = SAFETENSORS_HEADER_LEN_BYTES + header_len

    # manifest 里只保存分片文件名，不保存绝对路径，避免项目移动后路径失效。
    relative_file = shard_path.relative_to(model_dir).as_posix()

    # entries 保存当前分片中的所有 tensor 条目。
    entries: dict[str, dict[str, Any]] = {}

    # 遍历 header 中的每个条目；大多数条目是 tensor，也可能有 __metadata__。
    for tensor_name, metadata in header.items():
        # __metadata__ 是 safetensors 的文件级元信息，不是权重 tensor，必须跳过。
        if tensor_name == "__metadata__":
            continue

        # dtype 是 safetensors 记录的数据类型，例如 BF16、F16、F32。
        dtype = metadata["dtype"]

        # shape 是 tensor 形状，例如 [1408, 2048]。
        shape = metadata["shape"]

        # data_offsets 是相对数据区起点的 [start, end)，end 不包含在数据范围内。
        start, end = metadata["data_offsets"]

        # nbytes 是这个 tensor 在文件中占用的原始字节数。
        nbytes = end - start

        # 写入最终 manifest。
        # offset 使用文件内绝对偏移，后续可以直接：
        # open(file, "rb") -> seek(offset) -> read(nbytes)。
        entries[tensor_name] = {
            "file": relative_file,
            "offset": data_start + start,
            "nbytes": nbytes,
            "dtype": dtype,
            "shape": shape,
        }

    # 返回当前分片内所有 tensor 的位置信息。
    return entries


# 只有直接运行本文件时才执行 main()；被其他模块 import 时不会自动生成 manifest。
if __name__ == "__main__":
    main()
