from .rater import LLMRater


INITIAL = [
    "TheBloke/Llama-2-7b-Chat-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/Llama-2-13B-chat-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-7B-v1.3-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/vicuna-13b-v1.3.0-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-33B-GPTQ",  # use via exllama, on 24gb gpu
    "TheBloke/vicuna-7B-v1.5-GPTQ",  # use via exllama, on 8gb gpu
    "TheBloke/vicuna-13B-v1.5-GPTQ",  # use via exllama, on 24gb gpu
    #"TheBloke/Llama-2-70B-chat-GPTQ",  # use via exllama, on 80gb gpu
    "gpt-3.5-turbo-0613",  # use via openai api
    "gpt-4-0613",
    "gpt-4o",  # use via openai api
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    

]

TRANSFORMERS_MODELS = [
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "meta-llama/Meta-Llama-3-8B",
]

OLLAMA_MODELS= [
    'llama3.2',
    'gemma2',
    'mistral7b'
]


#SUPPORTED_MODELS = INITIAL + TRANSFORMERS_MODELS
SUPPORTED_MODELS = INITIAL + OLLAMA_MODELS

def load_LLM(name):
    print(f"name is {name}")
    if not name in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} not supported.")
    elif "gpt-3.5-turbo-0613" == name or "gpt-4-0613" in name or "gpt-4o" in name:
        from .openai_api import OpenAIModelAPI

        return OpenAIModelAPI(name)
    elif "GPTQ" in name and name in INITIAL:
        from .exllama import LLMExllama

        return LLMExllama(name)
    elif name in TRANSFORMERS_MODELS:
        from .std_transformers import Transformers

        return Transformers(name)
    elif name in OLLAMA_MODELS:

        from .ollama import OllamaLLM
        return OllamaLLM(name)