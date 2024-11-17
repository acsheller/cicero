import requests
from typing import Tuple, List, Dict
from .llm import LLM  # Assuming this is the existing class
import json

class OllamaLLM(LLM):
    def __init__(self, name: str, base_url: str = "http://ollama:11434/api/generate"):
        super().__init__(name)
        self.name = name
        self.base_url = base_url

    def request_rating_0_9(self, system_prompt: str, dialog: List[Dict[str, str]]) -> Tuple[str, str]:
        """Request a rating on a scale of 0-9."""
        prompt = self.encode(system_prompt, dialog)
        print(f"0_9 prompt is {prompt}")
        
        result = self._query_ollama(prompt)
        print(f"\n\n0_9 result is {result}\n\n")
        return result

    def request_rating_1_10(self, system_prompt: str, dialog: List[Dict[str, str]]) -> Tuple[str, str]:
        """Request a rating on a scale of 1-10."""
        prompt = self.encode(system_prompt, dialog)
        print(f"\n\n1_10 prompt is {prompt}")
        
        result = self._query_ollama(prompt)
        print(f"1_10 result is {result}")   
        return self._query_ollama(prompt)

    def request_rating_text(self, system_prompt: str, dialog: List[Dict[str, str]]) -> Tuple[str, str]:
        """Request a textual rating."""
        prompt = self.encode(system_prompt, dialog)
        print("request_rating_text")
        return self._query_ollama(prompt)

    def request_explanation(self, system_prompt: str, dialog: List[Dict[str, str]]) -> Tuple[str, str]:
        """Request an explanation."""
        prompt = self.encode(system_prompt, dialog)
        return self._query_ollama(prompt)

    def _query_ollama(self, prompt: str) -> Tuple[str, str]:
        """
        Send a prompt to Ollama and handle streaming responses.
        """
        payload = {"model": self.name, "prompt": prompt}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes

            # Collect the streamed response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)  # Use json.loads from the standard library
                    full_response += data.get("response", "")
                    if data.get("done"):
                        break
            print(f"\n --- full response is \n\n  {full_response} \n\n ---\n")
            return prompt, full_response

        except requests.ConnectionError as e:
            raise Exception(f"Connection error: {e}")
        except requests.RequestException as e:
            raise Exception(f"Request failed: {e}")
            