import fire
import sys
import torch
import json
import os
import numpy as np
import zlib
from llama import Llama
from langchain_openai import OpenAI
import torch.nn.functional as F

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

def compute_zlib_entropy(text):
    """
    Compute zlib entropy for a given text.
    
    Args:
        text (str): Input text.
    
    Returns:
        float: Zlib entropy value.
    """
    compressed = zlib.compress(text.encode('utf-8'))
    return len(compressed) / len(text) if len(text) > 0 else 0

def main(
        ckpt_dir: str,
        path: str,
        tokenizer_path: str = 'tokenizer.model',
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 256,
        max_batch_size: int = 1,
):
    """
    Main function to generate text and compute zlib entropy.

    Args:
        ckpt_dir (str): Model name or checkpoint directory.
        path (str): Input/output path.
        tokenizer_path (str): Tokenizer file.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling.
        max_seq_len (int): Maximum sequence length.
        max_gen_len (int): Maximum generated text length.
        max_batch_size (int): Maximum batch size.
    """

    print(path)
    generator = None
    llm = None

    # Detect if we are using LLaMA or GPT
    flag_llm = 'llama'
    if "gpt" in ckpt_dir:
        flag_llm = 'gpt'
        if ckpt_dir == "gpt":
            ckpt_dir = "gpt-3.5-turbo-instruct"
        llm = OpenAI(model=ckpt_dir, temperature=temperature, top_p=top_p, max_tokens=max_gen_len)
    else:
        generator = Llama.build(
            ckpt_dir='Model/' + ckpt_dir,
            tokenizer_path='Model/' + tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    # Load prompts
    print('Generating output...')
    with open(f"./Inputs&Outputs/{path}/prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())

    answers = []
    entropies = []

    for prompt in all_prompts:
        if flag_llm == "gpt":
            ans = llm.invoke(prompt)
        else:
            results = generator.text_completion(
                [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
            )
            ans = results[0]['generation']
        
        entropy = compute_zlib_entropy(ans)
        answers.append(ans)
        entropies.append(entropy)

    # Save generated outputs
    with open(f"./Inputs&Outputs/{path}/outputs-{ckpt_dir}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(answers))

    # Save zlib entropies
    with open(f"./Inputs&Outputs/{path}/zlib-entropies-{ckpt_dir}.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(entropies))

    print("Text generation and zlib entropy computation complete!")

if __name__ == "__main__":
    fire.Fire(main)
