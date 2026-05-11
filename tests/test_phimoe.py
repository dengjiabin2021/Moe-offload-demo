from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 加载模型（你下载的本地路径）
model_path = "D:\project_vscode\Moe-offload-demo\models\Phi-tiny-MoE-instruct"  # 改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # 👇 这行是核心！强制使用原生注意力，不依赖flash_attn
    attn_implementation="eager"
)


# 2. 推理
prompt = "你好，请介绍一下自己"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    cache_implementation="static",  # 必加！
    do_sample=False
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))