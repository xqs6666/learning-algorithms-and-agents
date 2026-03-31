from pyexpat import model
from openai import OpenAI

class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key,base_url=base_url)
        pass
    
    def generate(self,prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"
        
if __name__ == "__main__":
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key='sk-e6dad7d24b454fdab35a9041352e45dd'
    model_name ='qwen3.5-plus'
    
    model = OpenAICompatibleClient(model_name,api_key,base_url)
    answer = model.generate("你是小助手","你好")
    print(answer)