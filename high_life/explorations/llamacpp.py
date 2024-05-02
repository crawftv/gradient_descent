from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

model_url = "https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF/resolve/main/llama-3-8b-instruct-262k.Q4_0.gguf?download=true"

llamacpp = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    generate_kwargs={},
    max_new_tokens=256,
    context_window=32768,
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    # verbose=True,
)

if __name__ == "__main__":
    resp = llamacpp.complete("who is the president of the united states")
    print(resp)
