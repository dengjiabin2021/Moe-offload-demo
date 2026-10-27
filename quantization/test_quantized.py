from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "outputs" / "DeepSeek-V2-Lite-GPTQ-FP8"


def check_checkpoint(model_dir: Path) -> int:
    """
    检查量化配置、专家 FP8 权重和逐通道 scale。

    输入：量化模型目录。
    输出：通过检查的专家投影数量。
    功能：只读取根目录中的 safetensors 文件头，不把 offload 工作缓存当成最终模型。
    """

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    targets = set()
    for group in config["quantization_config"]["config_groups"].values():
        weights = group["weights"]
        activation = group["input_activations"]
        # 必须确实保存 W8A8 FP8 配置，不能把仅转换 dtype 或纯 W8 产物当成当前方案。
        assert weights["num_bits"] == 8 and weights["type"] == "float", "权重不是 FP8 配置"
        assert weights["strategy"] == "channel" and not weights["dynamic"], "权重不是逐通道静态量化"
        assert activation["num_bits"] == 8 and activation["type"] == "float", "输入不是 FP8 配置"
        assert activation["strategy"] == "token" and activation["dynamic"], "输入不是逐 token 动态量化"
        targets.update(group["targets"])

    tensors = {}
    for shard in model_dir.glob("*.safetensors"):
        with shard.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            tensors.update(json.loads(handle.read(header_size)))
    # 有配置但没有目标或没有最终分片，不能给出检查通过的结果。
    assert targets, "量化配置中没有目标投影"
    for target in sorted(targets):
        weight = tensors[f"{target}.weight"]
        scale = tensors[f"{target}.weight_scale"]
        # 每个目标必须是独立二维 FP8 投影，scale 必须与输出通道数对应。
        assert weight["dtype"] == "F8_E4M3" and len(weight["shape"]) == 2, target
        assert scale["dtype"] == "F32" and scale["shape"] == [weight["shape"][0], 1], target
    return len(targets)


def main() -> None:
    """
    加载量化模型并执行短文本生成。

    输入：量化模型目录、提示词、生成长度和 CPU/GPU 权重预算。
    输出：文件检查结果和实际生成的文本。
    功能：采用 BF16 反量化验证 FP8 产物能加载和推理，不验证原生 W8A8 内核或模型精度。
    """

    parser = argparse.ArgumentParser(description="检查 MoE FP8 模型并生成文本")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default="人工智能的主要应用包括")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory", default="6GiB")
    parser.add_argument("--cpu-memory", default="5GiB")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()

    count = check_checkpoint(model_dir)
    print(f"FP8 权重与 scale 检查通过：{count} 个投影。", flush=True)
    print("运行方式：FP8 权重反量化为 BF16，仅做加载与生成功能验证。", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    # 预算同时约束 CPU 和 GPU 驻留；放不下的权重由 Accelerate 卸载到独立工作目录。
    # 内置加载转换会先应用 FP8 scale，再把专家权重合并为原模型需要的布局。
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(model_dir / "test_offload"),
        offload_buffers=True,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )
    # 检查原始 logits，避免 logits processor 合法设置的 -inf 被误判为数值错误。
    assert output.logits and all(torch.isfinite(logits).all().item() for logits in output.logits), "生成出现 NaN 或 Inf"
    new_tokens = output.sequences.shape[1] - inputs["input_ids"].shape[1]
    assert new_tokens > 0, "模型没有生成新 token"
    print(f"加载与生成检查通过：生成 {new_tokens} 个 token。", flush=True)
    print(tokenizer.decode(output.sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
