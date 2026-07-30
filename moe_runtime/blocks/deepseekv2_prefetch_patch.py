from __future__ import annotations

from typing import Any

from torch import nn

from moe_runtime.prefetcher import DeepseekV2GatePrefetcher


def patch_deepseekv2_prefetch(
    *,
    model: nn.Module,
    routed_expert_layers: list[int],
    prefetcher: DeepseekV2GatePrefetcher,
) -> None:
    """
    为 DeepSeek-V2 模型注册 decode gate 预取 hook。

    输入：待修改模型、Adapter 提供的专家层列表和 DeepSeek-V2 预取器。
    输出：无。
    功能：模型级 hook 识别 decode 阶段，MoE block hook 在原 forward 前旁路提交未来专家预取。
    """

    def model_forward_pre_hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """
        根据输入序列长度和 KV Cache 长度判断当前生成阶段。

        输入：模型模块、位置参数和关键字参数。
        输出：无。
        功能：只有已有历史 KV 且本次序列长度为 1 时才标记为 decode，prefill 阶段保持关闭。
        """

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and len(args) > 4:
            inputs_embeds = args[4]

        sequence_length = 0
        if input_ids is not None:
            sequence_length = input_ids.shape[1]
        elif inputs_embeds is not None:
            sequence_length = inputs_embeds.shape[1]

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and len(args) > 3:
            past_key_values = args[3]

        past_length = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )
        prefetcher.set_decode_mode(sequence_length == 1 and past_length > 0)

    def build_moe_pre_hook(layer_idx: int):
        """
        创建绑定指定 DeepSeek-V2 MoE 层编号的 forward pre-hook。

        输入：当前 MoE 层编号。
        输出：可注册到该层 MoE block 的 hook 函数。
        功能：从原 block 输入中取得 gate hidden_states，并交给模型级 Prefetcher。
        """

        def moe_forward_pre_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
        ) -> None:
            """
            在原 DeepSeek-V2 MoE forward 前提交未来层预测。

            输入：当前 MoE block 和原始位置参数。
            输出：无。
            功能：旁路读取 hidden_states，不修改原 forward 的输入、输出和 routed expert 计算。
            """

            prefetcher.prefetch(
                current_layer_idx=layer_idx,
                hidden_states=args[0],
            )

        return moe_forward_pre_hook

    # 模型级 hook 维护本次 forward 的阶段，所有 DeepSeek-V2 MoE 层共享该状态。
    model.register_forward_pre_hook(model_forward_pre_hook, with_kwargs=True)

    # routed_expert_layers 已经跳过前置 dense 层和其他普通 MLP 层。
    for layer_idx in routed_expert_layers:
        model.model.layers[layer_idx].mlp.register_forward_pre_hook(
            build_moe_pre_hook(layer_idx)
        )
