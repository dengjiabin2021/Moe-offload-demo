from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from compressed_tensors.offload import disable_onloading
from compressed_tensors.quantization import preset_name_to_scheme
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modeling.moe.conversion_mappings import set_save_conversion_mapping
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


QUANTIZATION_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = QUANTIZATION_DIR.parent / "models" / "DeepSeek-V2-Lite"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def build_target_names(config: dict) -> list[str]:
    """
    按模型配置生成路由专家的投影名称。

    输入：原始 config.json 的内容。
    输出：需要量化的 gate、up、down 投影完整名称。
    功能：分别适配 DeepSeek-V2 和 Qwen2-MoE，排除稠密层、共享专家和 router。
    """

    if config["model_type"] == "deepseek_v2":
        experts = config["n_routed_experts"]
        # DeepSeek 的前几层可以是稠密层，专家层还受 moe_layer_freq 约束。
        layers = [
            index for index in range(config["num_hidden_layers"])
            if index >= config["first_k_dense_replace"]
            and index % config.get("moe_layer_freq", 1) == 0
        ]
    elif config["model_type"] == "qwen2_moe":
        experts = config["num_experts"]
        # Qwen 使用从 1 开始的 sparse step，并允许显式指定普通 MLP 层。
        layers = [
            index for index in range(config["num_hidden_layers"])
            if (index + 1) % config.get("decoder_sparse_step", 1) == 0
            and index not in config.get("mlp_only_layers", [])
        ]
    else:
        raise NotImplementedError(f"尚未适配模型：{config['model_type']}，请在 build_target_names 中增加分支。")

    return [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in layers
        for expert in range(experts)
        for projection in PROJECTIONS
    ]


def build_recipe(targets: list[str]) -> GPTQModifier:
    """
    构造路由专家的 GPTQ FP8 W8A8 配方。

    输入：需要量化的投影名称。
    输出：GPTQModifier 实例。
    功能：复用逐通道 W8、逐 token 动态 A8 预设，并用 FP32 保存 scale。
    """

    scheme = preset_name_to_scheme("FP8_DYNAMIC", targets)
    # 显式保留 FP32 scale，不随模型 BF16 dtype 降低缩放参数精度。
    scheme.weights.scale_dtype = torch.float32
    scheme.input_activations.scale_dtype = torch.float32
    return GPTQModifier(
        config_groups={"routed_experts": scheme},
        block_size=128,
        dampening_frac=0.01,
        actorder="static",
        # 每次只优化一个投影，避免同时堆叠多个专家的 Hessian 和临时权重。
        batched_quantization=1,
    )


def main() -> None:
    """
    读取校准文本，量化路由专家并保存模型。

    输入：命令行指定的模型目录、校准文件和内存预算。
    输出：固定目录中的 FP8 模型、tokenizer、许可证和校准日志。
    功能：沿用模型加载后线性化的流程，只按专家投影划分校准子图，不改模型源码。
    """

    parser = argparse.ArgumentParser(description="MoE 路由专家 GPTQ FP8 W8A8 量化")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibration-file", type=Path, default=QUANTIZATION_DIR / "calibration.json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--cpu-memory", default="5GiB", help="CPU 权重预算，不是进程总内存上限")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只检查模型配置和配方，不加载权重")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = (args.output_dir or QUANTIZATION_DIR / "outputs" / f"{model_dir.name}-GPTQ-FP8").resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    targets = build_target_names(config)
    # 配方在加载权重前构建，参数错误无需等完整模型加载后才发现。
    recipe = build_recipe(targets)
    print(f"原模型：{model_dir}", flush=True)
    print(f"输出目录：{output_dir}", flush=True)
    print(f"量化范围：{len(targets) // 3} 个路由专家，{len(targets)} 个投影", flush=True)
    print(f"方法：GPTQ FP8_DYNAMIC；CPU 权重预算：{args.cpu_memory}", flush=True)
    if args.dry_run:
        return

    # 保留原始模型；已有目录直接由 mkdir 报错，不自动覆盖或清理历史产物。
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("输出目录不能位于原模型目录中。")
    output_dir.mkdir(parents=True)
    set_seed(args.seed)
    # JSON 加载器同时支持 JSON 数组和 JSONL；只有抽样后的文本进入校准。
    dataset = load_dataset("json", data_files=str(args.calibration_file), split="train")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    dataset = Dataset.from_dict({"text": dataset[args.text_column]})
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)

    # 保留原来的加载方式：按 CPU 预算卸载到磁盘，再由库将融合专家拆为 Linear。
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            device_map="auto_offload",
            max_memory={"cpu": args.cpu_memory},
            offload_folder=str(output_dir / "offload"),
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        )
    model.eval()
    # 冻结磁盘参数的占位对象，不能因遍历参数而把整个模型重新读入内存。
    with disable_onloading():
        model.requires_grad_(False)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False

    print("开始 GPTQ 校准。", flush=True)
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=recipe,
        num_calibration_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        batch_size=1,
        shuffle_calibration_samples=False,
        pipeline="sequential",
        # 只把路由专家投影作为目标。attention 不是其祖先，会整块执行而不展开 FX 跟踪，
        # 因而不会进入 RoPE 的符号形状展开；也不需要把一整层专家同时搬进显存。
        sequential_targets=targets,
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=True,
        log_dir=str(output_dir / "logs"),
    )
    model.config.use_cache = original_use_cache

    print("开始保存量化模型。", flush=True)
    # 保留独立专家的二维权重名称，不把量化产物重新融合成三维张量。
    model.config.auto_map = {}
    set_save_conversion_mapping(model, [])
    model.save_pretrained(output_dir, save_compressed=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_dir)
    for license_path in model_dir.glob("LICENSE*"):
        shutil.copy2(license_path, output_dir / license_path.name)
    print(f"量化完成：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
