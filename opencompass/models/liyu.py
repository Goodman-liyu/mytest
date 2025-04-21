from ..base_api import BaseAPIModel
from typing import List
import requests
from time import sleep
import json


class ZhongzhuanAPI(BaseAPIModel):

    is_api: bool = True

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        query_per_second: int = 1,
        retry: int = 2,
        Baseurl: str = "https://api.claudeshop.top",
        Skey: str = "sk-HDpWybZEClV49CDgv6j3j2h0xra4dnzCujgT2il7427joO1b",
        **kwargs,
    ):

        meta_template = {}  # 需要根据实际情况定义元模板

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            retry=retry,
        )

        self.Baseurl = Baseurl
        self.Skey = Skey
        self.url = Baseurl + "/v1/chat/completions"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {Skey}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        inputs,
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        results = []

        for input_item in inputs:
            model_name = input_item.get("model_name", "gpt-3.5-turbo")
            system_prompt = input_item.get("system_prompt", "")
            user_prompt = input_item.get("user_prompt", "")

            retry_count = 0
            while retry_count <= self.retry:
                try:
                    payload = self._create_payload(model_name, system_prompt, user_prompt)
                    response = requests.post(self.url, headers=self.headers, data=payload)
                    response_data = response.json()
                    response.raise_for_status()  # 检查请求是否成功

                    content = response_data["choices"][0]["message"]["content"]
                    results.append(content)
                    break

                except requests.exceptions.RequestException as e:
                    print(f"请求失败: {e}")
                    retry_count += 1
                    sleep(1)  # 等待1秒后重试

                except KeyError as e:
                    print(f"解析响应失败: {e}")
                    retry_count += 1
                    sleep(1)  # 等待1秒后重试

                except Exception as e:
                    print(f"发生未知错误: {e}")
                    retry_count += 1
                    sleep(1)  # 等待1秒后重试

            if retry_count > self.retry:
                results.append("")  # 添加空字符串作为失败结果

        return results

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        # 这里需要实现获取令牌长度的方法
        # 简单实现可以是字符数除以4（粗略估计）
        return len(prompt) // 4

    def _create_payload(self, model_name, system_prompt, user_prompt, temperature=0.7):
        return json.dumps(
            {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
