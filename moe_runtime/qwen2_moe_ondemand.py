from __future__ import annotations

# gc 用来在专家权重用完后尽快释放 Python 引用关联的临时内存。
import gc
# json 用来读取 weight_manifest.json。
import json
# re 用来识别 routed expert 权重名，区分专家权重和非专家权重。
import re
# types.MethodType 用来把我们自定义的 forward 绑定到 transformers 原始 MoE 模块上。
import types
# inspect 用来检查当前 transformers 版本的 DecoderLayer 是否期待 MLP 返回 router logits。
import inspect
# Path 用来处理模型目录、manifest 路径等文件路径。
from pathlib import Path
# Any 用于 JSON manifest 条目的类型标注。
from typing import Any

# torch 是实际模型推理和 tensor 构造所必需的依赖。
import torch
# F.linear / F.softmax 等函数用于复现 Qwen2-MoE sparse MLP 的专家计算逻辑。
import torch.nn.functional as F
# nn.Module / nn.Parameter 用于替换模块和写入权重。
from torch import nn

# init_empty_weights 会让模型先构造在 meta 设备上，从而避免随机初始化占用内存。
from accelerate import init_empty_weights
# AutoConfig / AutoTokenizer 之外，这里只需要 AutoConfig 读取模型结构配置。
from transformers import AutoConfig
# ACT2FN 用来根据 config.hidden_act 找到 Qwen MLP 使用的激活函数，例如 silu。
from transformers.activations import ACT2FN
# 复用 transformers 官方 Qwen2MoeForCausalLM，避免自己重写 attention、norm、generate 等复杂逻辑。
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

# 复用前一步脚本里的 dtype 映射和尺寸校验，保证按 tensor_name 读取时解释方式一致。
from scripts.load_tensor_by_name import to_torch_dtype, validate_tensor_size


# Qwen1.5-MoE / Qwen2-MoE 的 routed expert 权重名形如：
# model.layers.0.mlp.experts.12.gate_proj.weight
# 这类权重不在初始化阶段加载，而是在 forward 命中专家时按需读取。
EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def load_qwen2_moe_with_ondemand_experts(
    model_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    empty_cache_after_expert: bool = True,
) -> Qwen2MoeForCausalLM:
    """加载 Qwen2-MoE 模型，其中 routed experts 在 forward 时按需从磁盘读取。"""

    # 模型目录中需要包含 config.json、tokenizer 文件、safetensors 分片和 weight_manifest.json。
    model_dir = Path(model_dir).resolve()

    # manifest 默认放在模型目录下；如果调用方传入自定义路径，就使用自定义路径。
    manifest_path = Path(manifest_path).resolve() if manifest_path else model_dir / "weight_manifest.json"

    # 目标设备通常是 cuda；这里转换成 torch.device，后续比较 device.type 更方便。
    device = torch.device(device)

    # 读取 manifest 到内存。manifest 只是元数据，不是权重缓存，体积很小。
    manifest = read_manifest(manifest_path)

    # 读取模型配置。Qwen1.5-MoE-A2.7B 的 architecture 是 Qwen2MoeForCausalLM。
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)

    # 使用 meta 初始化，避免 routed experts 或普通参数被随机初始化到 CPU/GPU。
    # 注意：新版 transformers 已经不再暴露 no_init_weights；init_empty_weights 本身就会把参数放到 meta 设备。
    with init_empty_weights():
        model = Qwen2MoeForCausalLM(config)

    # 在加载权重前替换 routed expert 容器，这样 expert 参数不会被创建成常驻 Parameter。
    install_ondemand_experts(
        model=model,
        model_dir=model_dir,
        manifest=manifest,
        device=device,
        empty_cache_after_expert=empty_cache_after_expert,
    )

    # 加载非 routed expert 权重，包括 attention、embedding、norm、lm_head、router gate、shared expert。
    load_resident_weights(
        model=model,
        model_dir=model_dir,
        manifest=manifest,
        device=device,
    )

    # demo 只做推理，因此切换到 eval 模式。
    model.eval()

    # 返回可直接调用 generate / forward 的模型。
    return model


class ManifestTensorLoader:
    """基于 weight_manifest.json，通过 tensor_name 精确读取单个 tensor。"""

    def __init__(self, *, model_dir: Path, manifest: dict[str, dict[str, Any]], device: torch.device) -> None:
        # 保存模型目录，用来和 manifest 中的分片文件名拼接成真实路径。
        self.model_dir = model_dir

        # 保存已读入内存的 manifest，避免每次专家命中都重新解析 JSON。
        self.manifest = manifest

        # 保存目标设备。专家权重读取后会移动到这个设备上参与计算。
        self.device = device

    def load(self, tensor_name: str) -> torch.Tensor:
        """按 tensor_name 从磁盘读取一个权重 tensor。"""

        # 从 manifest 中取出该 tensor 的文件名、offset、nbytes、dtype、shape。
        entry = self.manifest[tensor_name]

        # manifest 只保存分片文件名，所以这里拼出完整文件路径。
        shard_path = self.model_dir / entry["file"]

        # 如果分片不存在，说明模型目录不完整或 manifest 不是这个模型生成的。
        if not shard_path.is_file():
            raise FileNotFoundError(f"Safetensors shard does not exist: {shard_path}")

        # safetensors dtype 字符串，例如 BF16。
        dtype_name = entry["dtype"]

        # shape 是 tensor 原始形状，例如 [1408, 2048]。
        shape = entry["shape"]

        # offset 是文件内绝对字节偏移，可以直接 seek。
        offset = entry["offset"]

        # nbytes 是该 tensor 的精确字节长度，只读取这么多字节。
        nbytes = entry["nbytes"]

        # 读取前先校验 shape * dtype_size 是否等于 nbytes，避免按错误格式解释二进制数据。
        validate_tensor_size(dtype_name=dtype_name, shape=shape, nbytes=nbytes, tensor_name=tensor_name)

        # 把 safetensors dtype 转成 torch dtype。
        torch_dtype = to_torch_dtype(torch, dtype_name)

        # 只读取该 tensor 对应的字节范围，不读取整个分片文件。
        with shard_path.open("rb") as handle:
            handle.seek(offset)
            raw_bytes = handle.read(nbytes)

        # 如果读到的字节数不够，说明 offset/nbytes 或文件本身有问题。
        if len(raw_bytes) != nbytes:
            raise ValueError(
                f"Read {len(raw_bytes)} bytes for {tensor_name}, but manifest expects {nbytes} bytes."
            )

        # frombuffer 先在 CPU 上按 dtype 解释原始字节。
        # bytes 是只读 buffer，会触发 PyTorch 警告；bytearray 是可写 buffer，clone 后 tensor 拥有独立内存。
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype).clone()

        # reshape 恢复成原始矩阵形状。
        tensor = tensor.reshape(shape)

        # 最后移动到目标设备。对专家权重来说，这一步就是按需进入 GPU 显存。
        return tensor.to(self.device, non_blocking=True)


class OnDemandQwen2MoeExperts(nn.Module):
    """无常驻参数的 routed expert 容器，命中哪个专家就从磁盘读取哪个专家。"""

    def __init__(
        self,
        *,
        layer_idx: int,
        config: Any,
        tensor_loader: ManifestTensorLoader,
        empty_cache_after_expert: bool,
    ) -> None:
        super().__init__()

        # 当前 transformer layer 的编号，用来拼接 tensor_name。
        self.layer_idx = layer_idx

        # Qwen1.5-MoE-A2.7B 每层 60 个 routed experts。
        self.num_experts = int(config.num_experts)

        # 通过 manifest 精确读取 tensor 的小工具。
        self.tensor_loader = tensor_loader

        # 是否在每个专家计算结束后调用 torch.cuda.empty_cache()。
        self.empty_cache_after_expert = empty_cache_after_expert

        # routed expert 内部使用的激活函数，Qwen 配置中通常是 silu。
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """执行 routed expert 计算，专家权重用完即释放。"""

        # final_hidden_states 用来累加所有专家对 token 的输出。
        final_hidden_states = torch.zeros_like(hidden_states)

        # 只遍历本次 forward 实际命中的专家，避免读取未使用专家的权重。
        active_experts = torch.unique(selected_experts).tolist()

        # 逐个专家加载、计算、释放；这里没有缓存，也没有预取。
        for expert_idx in active_experts:
            # 找出哪些 token 的 top-k 路由选择了当前专家。
            token_idx, topk_idx = torch.where(selected_experts == expert_idx)

            # 理论上 active_experts 都有命中 token，这里保留防御性检查。
            if token_idx.numel() == 0:
                continue

            # 当前专家只处理被路由到它的 token。
            current_state = hidden_states[token_idx]

            # 按需从磁盘读取当前专家的三块权重到 GPU。
            gate_w, up_w, down_w = self.load_expert_weights(int(expert_idx))

            # Qwen MoE expert 的计算：down_proj(act(gate_proj(x)) * up_proj(x))。
            expert_output = F.linear(
                self.act_fn(F.linear(current_state, gate_w)) * F.linear(current_state, up_w),
                down_w,
            )

            # 乘上 router 给当前 expert 的 routing weight。
            expert_output = expert_output * routing_weights[token_idx, topk_idx, None]

            # 多个专家的输出累加回对应 token 位置。
            final_hidden_states.index_add_(0, token_idx, expert_output.to(final_hidden_states.dtype))

            # 删除临时引用，确保本轮专家权重不再被 Python 持有。
            del gate_w, up_w, down_w, current_state, expert_output

            # 立即释放缓存分配器中可回收的显存；这会牺牲速度，但符合“先不缓存”的 demo 目标。
            self.release_after_expert()

        # 返回所有命中专家累加后的结果。
        return final_hidden_states

    def load_expert_weights(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取某层某个 expert 的 gate/up/down 三个矩阵。"""

        # Qwen expert 有三个无 bias 线性层：gate_proj、up_proj、down_proj。
        return (
            self.load_expert_weight(expert_idx, "gate_proj"),
            self.load_expert_weight(expert_idx, "up_proj"),
            self.load_expert_weight(expert_idx, "down_proj"),
        )

    def load_expert_weight(self, expert_idx: int, proj_name: str) -> torch.Tensor:
        """按 Qwen 权重命名规则拼 tensor_name 并读取。"""

        # tensor_name 必须和 manifest 中的 key 完全一致。
        tensor_name = f"model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"

        # 通过 manifest 精确读取该 tensor。
        return self.tensor_loader.load(tensor_name)

    def release_after_expert(self) -> None:
        """专家计算后尽快释放临时 CPU/GPU 内存。"""

        # 先触发 Python 垃圾回收，释放 raw bytes / CPU tensor 等临时对象。
        gc.collect()

        # 如果目标设备是 CUDA，并且用户没有关闭 empty_cache，就清理 PyTorch CUDA cache。
        if self.empty_cache_after_expert and self.tensor_loader.device.type == "cuda":
            torch.cuda.empty_cache()


def sparse_moe_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """替换 transformers 原始 sparse MoE forward，接入磁盘按需专家。"""

    # 原始 hidden_states 是 [batch, seq, hidden_dim]。
    batch_size, sequence_length, hidden_dim = hidden_states.shape

    # router 和 expert 计算都按 token 展平处理。
    flat_states = hidden_states.reshape(-1, hidden_dim)

    # router gate 常驻 GPU。不同 transformers 版本的 gate 返回值不一样：
    # 新版直接返回 (_, routing_weights, selected_experts)，旧版可能只返回 router_logits。
    gate_output = self.gate(flat_states)

    # transformers 5.x 的 Qwen2-MoE gate 已经在内部完成 softmax/topk。
    if isinstance(gate_output, tuple):
        router_logits, routing_weights, selected_experts = gate_output

    # 旧版 transformers 中 gate 可能只返回 logits，此时在这里复现原始 softmax/topk 逻辑。
    else:
        router_logits = gate_output

        # 把 logits 转成概率；这里保持 transformers 官方实现的 float softmax 行为。
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 每个 token 只选择 top_k 个 expert，Qwen1.5-MoE-A2.7B 通常是 4 个。
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # 部分 MoE 配置会要求 top-k 概率重新归一化；Qwen 配置里 norm_topk_prob 通常是 false。
        if getattr(self, "norm_topk_prob", False):
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # routing weight 转回模型 dtype，避免后续混合 dtype 计算产生不必要开销。
        routing_weights = routing_weights.to(flat_states.dtype)

    # routed experts 走磁盘按需加载路径。
    expert_output = self.experts(flat_states, selected_experts, routing_weights)

    # shared expert 是非 routed expert，初始化时已经常驻 GPU。
    shared_output = self.shared_expert(flat_states)

    # shared_expert_gate 也是常驻 GPU 参数，用 sigmoid 控制 shared expert 输出强度。
    shared_output = F.sigmoid(self.shared_expert_gate(flat_states)) * shared_output

    # routed experts 输出与 shared expert 输出相加，再恢复 [batch, seq, hidden_dim]。
    output = (expert_output + shared_output).reshape(batch_size, sequence_length, hidden_dim)

    # 旧版 transformers 的 DecoderLayer 可能期待 (hidden_states, router_logits)；
    # 新版 transformers 5.x 只期待 hidden_states。安装替换时会设置这个兼容标记。
    if getattr(self, "_ondemand_returns_router_logits", False):
        return output, router_logits
    return output


def install_ondemand_experts(
    *,
    model: Qwen2MoeForCausalLM,
    model_dir: Path,
    manifest: dict[str, dict[str, Any]],
    device: torch.device,
    empty_cache_after_expert: bool,
) -> None:
    """把每层原始 experts 替换成无参数的按需加载 experts。"""

    # 创建共享的 manifest loader；它只缓存元数据，不缓存任何权重 tensor。
    tensor_loader = ManifestTensorLoader(model_dir=model_dir, manifest=manifest, device=device)

    # 遍历 transformer layers，找到 sparse MoE MLP。
    for layer_idx, layer in enumerate(model.model.layers):
        # Qwen sparse layer 的 MLP 通常挂在 layer.mlp。
        mlp = getattr(layer, "mlp", None)

        # 如果某层没有 experts，说明它不是 sparse MoE 层，跳过即可。
        if mlp is None or not hasattr(mlp, "experts"):
            continue

        # 替换 experts 容器：新容器没有 Parameter，因此专家权重不会常驻内存。
        mlp.experts = OnDemandQwen2MoeExperts(
            layer_idx=layer_idx,
            config=model.config,
            tensor_loader=tensor_loader,
            empty_cache_after_expert=empty_cache_after_expert,
        )

        # 记录当前 transformers 版本的上层 DecoderLayer 是否期待 MLP 返回 router logits。
        mlp._ondemand_returns_router_logits = decoder_layer_expects_router_logits(layer)

        # 绑定新的 sparse MoE forward，让 router 输出接入我们的磁盘专家容器。
        mlp.forward = types.MethodType(sparse_moe_forward, mlp)


def decoder_layer_expects_router_logits(layer: nn.Module) -> bool:
    """判断当前 transformers 版本的 DecoderLayer 是否把 mlp 返回值解包成 hidden/router_logits。"""

    try:
        source = inspect.getsource(layer.forward)
    except OSError:
        # 如果拿不到源码，默认按新版 transformers 5.x 行为处理：MLP 只返回 hidden_states。
        return False

    # 旧版源码通常包含类似：hidden_states, router_logits = self.mlp(hidden_states)。
    return "router_logits" in source and "self.mlp" in source


def load_resident_weights(
    *,
    model: Qwen2MoeForCausalLM,
    model_dir: Path,
    manifest: dict[str, dict[str, Any]],
    device: torch.device,
) -> None:
    """初始化阶段加载所有非 routed expert 权重，并让它们常驻目标设备。"""

    # 这里的 loader 也只是读取单个 tensor，不缓存权重。
    tensor_loader = ManifestTensorLoader(model_dir=model_dir, manifest=manifest, device=device)

    # 逐个遍历 manifest 中的权重名。
    for tensor_name in manifest:
        # routed expert 权重跳过，等待 forward 命中时按需加载。
        if is_routed_expert_weight(tensor_name):
            continue

        # 读取非专家权重到目标设备；shared_expert 不匹配 routed expert 正则，因此会在这里加载。
        tensor = tensor_loader.load(tensor_name)

        # 把 tensor 写回 meta 模型中对应的 Parameter / buffer。
        set_module_tensor(model, tensor_name, tensor)


def set_module_tensor(model: nn.Module, tensor_name: str, tensor: torch.Tensor) -> None:
    """把一个真实 tensor 写入 meta 初始化后的模型参数位置。"""

    # 找到 tensor_name 对应的父模块和属性名。
    parent, attr_name = resolve_parent_module(model, tensor_name)

    # 读取原位置的对象，通常是 meta Parameter。
    current_value = getattr(parent, attr_name)

    # 如果原对象是 Parameter，就用真实 tensor 包成新的 Parameter。
    if isinstance(current_value, nn.Parameter):
        setattr(parent, attr_name, nn.Parameter(tensor, requires_grad=current_value.requires_grad))
        return

    # 如果原对象是 buffer，就写入 _buffers。
    if isinstance(current_value, torch.Tensor):
        parent._buffers[attr_name] = tensor
        return

    # 其他类型说明路径解析或模型结构不符合预期。
    raise TypeError(f"Cannot assign tensor to {tensor_name}: found {type(current_value)!r}")


def resolve_parent_module(model: nn.Module, tensor_name: str) -> tuple[nn.Module, str]:
    """根据 state_dict 风格的 tensor_name 找到父模块和最后一级属性名。"""

    # 例如 model.layers.0.self_attn.q_proj.weight 会切成多个路径片段。
    parts = tensor_name.split(".")

    # 从根模型开始逐级向下访问。
    parent: Any = model

    # 最后一段是 weight/bias，不参与父模块查找。
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # 返回父模块和最后的属性名，例如 (q_proj, "weight")。
    return parent, parts[-1]


def read_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """读取 weight_manifest.json。"""

    # manifest 不存在时，提示先运行切片脚本。
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}. "
            "请先运行 scripts/build_safetensors_manifest.py 生成 weight_manifest.json。"
        )

    # manifest 顶层应该是 {tensor_name: {file, offset, nbytes, dtype, shape}}。
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_routed_expert_weight(tensor_name: str) -> bool:
    """判断一个权重名是否属于 routed expert。"""

    # 匹配 model.layers.*.mlp.experts.*.(gate/up/down)_proj.weight。
    return EXPERT_WEIGHT_RE.match(tensor_name) is not None
