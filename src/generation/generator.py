# src/generation/generator.py
import yaml

def load_config(config_path="configs/configs.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_answer(prompt: str, config_path="configs/configs.yaml") -> str:
    """
    根据组装好的 Prompt 调用大模型生成回答。
    目前支持 dummy（测试用）和 openai 模式。
    """
    config = load_config(config_path)
    backend = config.get("models", {}).get("generator_backend", "dummy")
    
    if backend == "dummy":
        # 假装大模型思考了半天...
        return "[Dummy 模式返回]: 这是一个测试回答。我收到了你的 Prompt，但我现在是个假模型，所以只能给你返回这句废话。"
    
    elif backend == "openai":
        # 等你有了 API Key 再把这里补充完整
        from openai import OpenAI
        # client = OpenAI(api_key="your_api_key_here")
        # response = client.chat.completions.create(...)
        # return response.choices[0].message.content
        return "OpenAI 接口还没写完呢，先用 dummy 跑吧！"
    
    else:
        return f"不支持的 backend: {backend}"

# 测试一下
if __name__ == "__main__":
    from prompt_builder import build_rag_prompt
    
    test_prompt = build_rag_prompt("什么是增值税？", "增值税是以商品在流转过程中产生的增值额作为计税依据而征收的一种流转税。")
    answer = generate_answer(test_prompt)
    
    print("\n--- 大模型回答 ---")
    print(answer)