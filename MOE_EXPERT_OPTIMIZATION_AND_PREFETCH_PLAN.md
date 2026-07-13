# MoE Single Expert Operator and Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最小改动实现两个优化：单 expert 计算合并，以及异步 expert 预取。

**Architecture:** 不做多 expert 合并计算，不重写整体调度体系。算子优化只替换单个 expert 内部的 `gate/up/down` 计算路径；预取只给 async CacheManager 增加低优先级 disk->CPU 提前加载，不改变 block 的 demand 计算逻辑。

**Tech Stack:** PyTorch、CUDA extension、Triton、CUDA event 计时、现有 async CacheManager 和 async block。

---

## 1. 范围收缩

### 做

- 单 expert 计算优化。
- `gate_proj` 和 `up_proj` 合并计算。
- 分别设计 PyTorch、Triton、CUDA 三种实现路径。
- 编写独立测试文件，构造随机输入，比较优化前后：
  - 数值误差。
  - decode 场景耗时。
  - prefill 场景耗时。
- 实现简单预取：根据上一 token 的路由结果，提前把下一层可能用到的 expert 从磁盘读到 CPU cache。

### 不做

- 不做多 expert grouped GEMM。
- 不做跨层大规模调度重构。
- 不把 profiler 指标写进 runtime 逻辑文件。
- 不把整个 CPU cache 改成 pinned memory。
- 不新增复杂状态，例如 prefetch protected、prefetch ready、cancelled、evicted。
- 不让预取抢占当前 demand request。

---

## 2. 单 Expert 算子优化方案

### 2.1 当前计算

当前 block 内单 expert 计算逻辑等价于：

```python
gate = F.linear(x, gate_weight)
up = F.linear(x, up_weight)
hidden = act(gate) * up
out = F.linear(hidden, down_weight)
out = out * router_weight
final.index_add_(0, token_idx, out)
```

其中一次 expert 需要：

- `gate_proj` GEMM/GEMV。
- `up_proj` GEMM/GEMV。
- `act(gate) * up`。
- `down_proj` GEMM/GEMV。
- router weight 乘法。
- `index_add_` 写回。

这次优化只处理单 expert 内部：

- 合并 `gate_proj` 和 `up_proj`。
- 尽量减少 kernel launch。
- 不改变 expert 顺序。
- 不改变 router 选择。
- 不改变 `index_add_` 的语义。

### 2.2 最小文件设计

只新增和修改这些文件：

- Create: `moe_runtime/ops/single_expert.py`
  - 放 PyTorch reference、PyTorch gate/up 合并版、Triton 调用入口、CUDA 调用入口。
- Create: `moe_runtime/ops/triton_single_expert.py`
  - 放 Triton kernel。没有 Triton 时直接报清晰错误。
- Create: `moe_runtime/ops/cuda_single_expert.cpp`
  - PyTorch extension C++ 绑定。
- Create: `moe_runtime/ops/cuda_single_expert_kernel.cu`
  - CUDA kernel。
- Create: `test/test_single_expert_op.py`
  - 构造输入、校验数值、比较性能。
- Modify: `moe_runtime/blocks/async_qwen2_moe_block.py`
  - 只替换 `_compute_ready_expert()` 里的单 expert 计算调用。
- Modify: `moe_runtime/blocks/async_deepseekv2_moe_block.py`
  - 同 Qwen。
- Modify: `moe_runtime/config.py`
  - 增加 `expert_compute_backend`，默认 `"torch"`。

不新增单独 benchmark 目录，不新增实验目录，不新增大规模 profiler 逻辑。

### 2.3 Backend 选择

`RuntimeConfig` 增加：

```python
expert_compute_backend: str = "torch"
```

合法值：

- `"torch"`：现有 3 次 `F.linear`，作为 reference。
- `"torch_gate_up"`：PyTorch 版 gate/up 合并，一次 `F.linear(x, gate_up_weight)`。
- `"triton_gate_up"`：Triton 版 gate/up + activation 合并。
- `"cuda_gate_up"`：CUDA extension 版 gate/up + activation 合并。

默认保持 `"torch"`。只有测试证明收益稳定，再手动切换。

---

## 3. PyTorch 合并版

### 3.1 核心思路

把两个权重：

```python
gate_weight: [intermediate, hidden]
up_weight:   [intermediate, hidden]
```

拼成一个连续权重：

```python
gate_up_weight: [2 * intermediate, hidden]
```

计算改成：

```python
gate_up = F.linear(x, gate_up_weight)
gate, up = gate_up.chunk(2, dim=-1)
hidden = act(gate) * up
out = F.linear(hidden, down_weight)
```

这样单 expert 从 3 次 linear 变成 2 次 linear。

### 3.2 权重布局

第一版不要在 forward 里临时 `torch.cat([gate, up])`，否则会引入额外分配和拷贝。

更合理的方式是在 CacheManager 装载 expert 时就把 gate/up 写成连续布局：

```text
single expert buffer:
| gate_weight | up_weight | down_weight |
```

`_view_gpu_slot()` 返回：

```python
{
    "gate_up": gate_up_view,
    "gate": gate_view,
    "up": up_view,
    "down": down_view,
}
```

这样 `"torch"` 和 `"torch_gate_up"` 可以共用同一份 buffer。

### 3.3 接入点

只改 async block 的 `_compute_ready_expert()`：

```python
expert_tensor_dict = self.cache_manager.get_ready_expert(...)
backend = self.cache_manager.config.expert_compute_backend

if backend == "torch_gate_up":
    current_hidden_states = single_expert_torch_gate_up(...)
elif backend == "triton_gate_up":
    current_hidden_states = single_expert_triton_gate_up(...)
elif backend == "cuda_gate_up":
    current_hidden_states = single_expert_cuda_gate_up(...)
else:
    current_hidden_states = single_expert_torch(...)
```

其他 ready/pending/request/release 逻辑不动。

---

## 4. Triton 版本

### 4.1 目标

Triton 第一版只做：

```text
x + gate_up_weight -> hidden = act(gate) * up
```

也就是合并：

- gate/up 投影。
- activation。
- gate/up 乘法。

`down_proj` 先继续用 PyTorch `F.linear(hidden, down_weight)`。

原因：

- 完整融合 `down_proj` 会把两个 GEMM 强行塞进一个 kernel，复杂度高，寄存器和 shared memory 压力大。
- 当前目标是单 expert 计算合并，不做大规模 kernel 重写。
- 先验证 gate/up 合并能否降低 decode 小 batch 下的 launch 和读写开销。

### 4.2 输入输出

Triton 函数接口：

```python
def single_expert_triton_gate_up(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    router_weight: torch.Tensor,
    act_name: str,
) -> torch.Tensor:
    ...
```

输入 shape：

```text
x:              [num_tokens_for_expert, hidden]
gate_up_weight: [2 * intermediate, hidden]
down_weight:    [hidden, intermediate]
router_weight:  [num_tokens_for_expert, 1]
```

输出：

```text
out: [num_tokens_for_expert, hidden]
```

### 4.3 Triton kernel 分两步

第一步 Triton：

```text
gate = x @ gate_weight.T
up   = x @ up_weight.T
hidden = silu(gate) * up
```

第二步 PyTorch：

```python
out = F.linear(hidden, down_weight)
out = out * router_weight
```

如果第一版 Triton 没有优势，不继续做 Triton down kernel。

---

## 5. CUDA 版本

### 5.1 目标

CUDA 第一版也只做单 expert：

```text
x + gate_up_weight -> hidden = act(gate) * up
```

`down_proj` 继续用 PyTorch。

### 5.2 为什么 CUDA 和 Triton 都要做

- Triton 开发快，方便验证 tile 策略。
- CUDA 控制更细，可以针对 decode 的 `num_tokens_for_expert = 1` 做 GEMV 优化。
- 如果 Triton 在 Windows 环境不可用，CUDA extension 仍然可以作为可控路径。

### 5.3 CUDA extension 接口

Python 调用：

```python
hidden = cuda_gate_up_swiglu(x, gate_up_weight)
out = F.linear(hidden, down_weight)
out = out * router_weight
```

C++ binding：

```cpp
torch::Tensor gate_up_swiglu_cuda(torch::Tensor x, torch::Tensor gate_up_weight);
```

CUDA kernel 第一版只支持：

- CUDA tensor。
- fp16。
- contiguous input。
- `act_fn = silu`。

这是为了控制实现复杂度。Qwen 当前就是 SwiGLU；其他 activation 后续再加。

---

## 6. 单 Expert 测试设计

### 6.1 测试文件

Create: `test/test_single_expert_op.py`

测试内容只做三件事：

1. 构造随机输入。
2. 比较 reference 和优化版数值。
3. 比较 reference 和优化版耗时。

### 6.2 测试输入

覆盖两类场景：

```python
CASES = [
    # decode：一个 expert 通常只处理少量 token
    {"tokens": 1, "hidden": 2048, "intermediate": 1408},
    {"tokens": 4, "hidden": 2048, "intermediate": 1408},

    # prefill：一个 expert 可能处理多个 token
    {"tokens": 27, "hidden": 2048, "intermediate": 1408},
    {"tokens": 128, "hidden": 2048, "intermediate": 1408},
]
```

构造：

```python
x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
gate = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)
up = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)
gate_up = torch.cat([gate, up], dim=0).contiguous()
down = torch.randn(hidden, intermediate, device="cuda", dtype=torch.float16)
router_weight = torch.randn(tokens, 1, device="cuda", dtype=torch.float16)
```

### 6.3 正确性判断

reference：

```python
expected = single_expert_torch(
    x=x,
    gate_weight=gate,
    up_weight=up,
    down_weight=down,
    router_weight=router_weight,
)
```

优化版：

```python
actual = single_expert_torch_gate_up(
    x=x,
    gate_up_weight=gate_up,
    down_weight=down,
    router_weight=router_weight,
)
```

误差标准：

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

Triton/CUDA 版本同样和 reference 比。

### 6.4 性能测试

使用 CUDA event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(warmup):
    fn()

torch.cuda.synchronize()
start.record()
for _ in range(repeat):
    fn()
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end) / repeat
```

输出格式：

```text
tokens=1 hidden=2048 intermediate=1408
torch:          0.XXX ms
torch_gate_up:  0.XXX ms  speedup=X.XXx
triton_gate_up: 0.XXX ms  speedup=X.XXx
cuda_gate_up:   0.XXX ms  speedup=X.XXx
```

运行命令：

```powershell
python test/test_single_expert_op.py --backend torch_gate_up
python test/test_single_expert_op.py --backend triton_gate_up
python test/test_single_expert_op.py --backend cuda_gate_up
```

验收：

- 所有 backend 数值对齐 reference。
- `tokens=1` 时重点看 decode 收益。
- `tokens=27/128` 时确认 prefill 不明显退化。

---

## 7. Expert 预取方案

### 7.1 目标

预取只解决一个问题：

```text
decode 阶段下一个 layer 需要 expert 时，尽量不要再从磁盘开始等。
```

第一版只预取到 CPU cache。

原因：

- GPU cache 很小，错误预取容易污染 GPU。
- CPU cache 比 GPU cache 大，预取到 CPU 后，真正 demand 到来时只需要 H2D。
- 当前 async 逻辑已经能把 H2D 和计算重叠。

### 7.2 预取信息来源

使用上一 token 的路由历史。

例子：

```text
当前正在计算 layer 8。
上一 token 在 layer 9 命中过 experts [3, 12, 21, 57]。
那么当前 token 计算 layer 8 时，提前把 layer 9 的 [3, 12, 21, 57] 读到 CPU。
```

这个策略简单，可测，不需要预测模型，也不需要改 router。

### 7.3 Prefetcher 接口

Create: `moe_runtime/prefetcher.py`

只需要一个类：

```python
class AsyncExpertPrefetcher:
    def before_layer(self, layer_idx: int) -> None:
        ...

    def observe_route(self, layer_idx: int, expert_ids: list[int]) -> None:
        ...
```

block 调用位置：

```python
if self.prefetcher is not None:
    self.prefetcher.before_layer(self.layer_idx)

# 算出 active_experts 后
if self.prefetcher is not None:
    self.prefetcher.observe_route(self.layer_idx, active_experts)
```

### 7.4 CacheManager 预取接口

在 `asyncExpertCacheManager` 增加：

```python
def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
    ...
```

语义：

- 如果 expert 已在 CPU/GPU，跳过。
- 如果 expert 已经有 demand request，跳过。
- 如果 expert 已经有 prefetch request，跳过。
- 否则加入低优先级 prefetch 队列。

### 7.5 最小状态

只新增三类结构：

```python
self._prefetch_requests: set[tuple[int, int]]
self._prefetch_waiting_cpu: deque[tuple[int, int]]
self._prefetch_loading_cpu: set[tuple[int, int]]
```

不新增 GPU 预取队列。

### 7.6 调度规则

原有 demand 调度优先级不变：

```text
waiting_cpu -> loading_cpu -> waiting_gpu -> loading_gpu -> ready
```

预取只在磁盘 worker 有空闲时执行：

```text
先调度 demand disk read。
如果 disk worker 还有空闲，再调度 prefetch disk read。
```

prefetch 磁盘读取完成后：

```text
CPU slot 状态 = used
不加入 ready
不进入 waiting_gpu
不 protected
```

当后续 demand request 到来：

- 如果 CPU 命中，走现有 CPU hit -> waiting_gpu。
- 如果 prefetch 正在 loading，直接把这个 key 提升为 demand request。

第一版可以只实现 CPU-only prefetch。GPU-free prefetch 放到后续实验，不写进第一版代码。

---

## 8. 预取测试设计

### 8.1 单元测试

Create: `test/test_async_prefetcher.py`

测试点：

- `observe_route()` 能记录每层上一 token 命中 expert。
- `before_layer()` 会预取下一层上一 token 的 expert。
- `prefetch_experts()` 不重复提交同一个 expert。
- prefetch 完成后 CPU slot 是 `used`。
- demand request 可以使用 prefetch 得到的 CPU slot。

### 8.2 端到端测试

直接扩展现有：

- Modify: `test/test_qwen_serial_async_benchmark.py`
- Modify: `test/test_async_profiler.py`

增加参数：

```powershell
--prefetch-policy none
--prefetch-policy last_token_next_layer_cpu
```

对比命令：

```powershell
python test/test_qwen_serial_async_benchmark.py --prefetch-policy none
python test/test_qwen_serial_async_benchmark.py --prefetch-policy last_token_next_layer_cpu
python test/test_async_profiler.py --prefetch-policy last_token_next_layer_cpu
```

重点看：

- decode 时间。
- decode 阶段 `wait_any_ready`。
- decode 阶段 CPU hit / disk miss。
- 输出文本是否一致。

---

## 9. 实施顺序

### Step 1: 单 expert PyTorch 合并版

- [ ] 新增 `moe_runtime/ops/single_expert.py`。
- [ ] 新增 `test/test_single_expert_op.py`。
- [ ] 先实现 `"torch"` 和 `"torch_gate_up"`。
- [ ] 跑随机输入数值测试。
- [ ] 跑 decode/prefill 性能测试。
- [ ] 如果 `torch_gate_up` 没有收益，先不接入 block。

### Step 2: Triton 版本

- [ ] 新增 `moe_runtime/ops/triton_single_expert.py`。
- [ ] 实现 gate/up + silu + multiply。
- [ ] `down_proj` 继续用 PyTorch。
- [ ] 在 `test/test_single_expert_op.py` 里加入 `--backend triton_gate_up`。
- [ ] 比较 `tokens=1/4/27/128`。

### Step 3: CUDA 版本

- [ ] 新增 `moe_runtime/ops/cuda_single_expert.cpp`。
- [ ] 新增 `moe_runtime/ops/cuda_single_expert_kernel.cu`。
- [ ] 实现 fp16 contiguous 的 gate/up + silu + multiply。
- [ ] `down_proj` 继续用 PyTorch。
- [ ] 在 `test/test_single_expert_op.py` 里加入 `--backend cuda_gate_up`。
- [ ] 比较 CUDA 和 Triton 版本。

### Step 4: 接入 async block

- [ ] 在 `RuntimeConfig` 增加 `expert_compute_backend`。
- [ ] 修改 `async_qwen2_moe_block.py` 的 `_compute_ready_expert()`。
- [ ] 修改 `async_deepseekv2_moe_block.py` 的 `_compute_ready_expert()`。
- [ ] 用 `test_qwen_serial_async_benchmark.py` 验证输出一致。
- [ ] 用 `test_async_profiler.py` 验证 prefill/decode 时间。

### Step 5: CPU-only expert 预取

- [ ] 新增 `moe_runtime/prefetcher.py`。
- [ ] 在 `runtime_engine.py` 根据 `prefetch_policy` 创建 prefetcher。
- [ ] 在 async Qwen/DeepSeek block 调用 `before_layer()` 和 `observe_route()`。
- [ ] 在 async CacheManager 增加 CPU-only `prefetch_experts()`。
- [ ] demand 队列优先，prefetch 队列只用空闲 disk worker。
- [ ] prefetch 完成后 CPU slot 标记为 `used`。

### Step 6: 预取效果验证

- [ ] 运行 `--prefetch-policy none`。
- [ ] 运行 `--prefetch-policy last_token_next_layer_cpu`。
- [ ] 比较 decode 时间和 `wait_any_ready`。
- [ ] 如果 prefetch 让 decode 变慢，保持默认 `none`，再考虑限制每层预取 expert 数量。

---

## 10. 验收标准

### 算子优化

- 数值一致：

```text
torch_gate_up / triton_gate_up / cuda_gate_up 与 torch reference 误差在 rtol=1e-2, atol=1e-2 内。
```

- 性能输出清晰：

```text
每个 backend 都打印 tokens=1/4/27/128 的平均耗时和 speedup。
```

- 接入 runtime 后：

```text
output same: True
```

### 预取

- 不改变输出：

```text
prefetch_policy=none 和 last_token_next_layer_cpu 输出一致。
```

- 不干扰 demand：

```text
demand request 优先级始终高于 prefetch。
```

- 判断是否值得保留：

```text
decode 时间下降，或者 decode wait_any_ready 明显下降。
```

如果 decode 变慢，预取功能保留为可选策略，默认关闭。
