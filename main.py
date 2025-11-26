from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "download/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # CPU 使用 float32
    low_cpu_mem_usage=False  # 禁用 low_cpu_mem_usage 以避免需要 accelerate
)
model = model.to("cpu")  # 明确指定使用 CPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=5
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
