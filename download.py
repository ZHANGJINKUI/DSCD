from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型名称
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 下载并保存模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 保存到本地
tokenizer.save_pretrained("./local_llama2_tokenizer")
model.save_pretrained("./local_llama2_model")