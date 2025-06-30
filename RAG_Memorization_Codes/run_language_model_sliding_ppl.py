import fire
import sys
import torch
import json
import os
import numpy as np
from llama import Llama
from langchain_openai import OpenAI
import torch.nn.functional as F

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

def compute_sliding_window_perplexity(generator, prompt, max_gen_len, temperature, top_p, flag_llm, window_size=50):
    """
    Compute sliding window perplexity manually.

    Args:
        generator: The LLM generator object (LLaMA or OpenAI).
        prompt (str): The input text prompt.
        max_gen_len (int): Maximum length of the generated text.
        temperature (float): Temperature for randomness.
        top_p (float): Top-p sampling.
        flag_llm (str): Either "llama" or "gpt".
        window_size (int): The number of tokens in each perplexity window.

    Returns:
        tuple: (generated text, sliding window perplexity value)
    """
    if flag_llm == "gpt":
        results = generator.invoke(prompt)
        generated_text = results
        return generated_text, None  # OpenAI models don't return logits

    # Generate text from LLaMA
    results = generator.text_completion(
        [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
    )
    generated_text = results[0]['generation']

    # Tokenize generated text
    tokens = generator.tokenizer.encode(generated_text, bos=True, eos=True)
    input_ids = torch.tensor([tokens], dtype=torch.long).cuda()

    # Forward pass to get logits
    with torch.no_grad():
        output = generator.model(input_ids, start_pos=0)
        logits = output if isinstance(output, torch.Tensor) else output.logits

    # Compute log probabilities
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Sliding window perplexity computation
    num_tokens = token_log_probs.shape[1]
    sliding_ppls = []

    for i in range(0, num_tokens - window_size + 1):
        window_log_probs = token_log_probs[:, i:i + window_size]
        avg_log_prob = window_log_probs.mean().item()
        sliding_ppl = np.exp(-avg_log_prob)
        sliding_ppls.append(sliding_ppl)

    overall_ppl = np.mean(sliding_ppls) if sliding_ppls else float('inf')
    return generated_text, overall_ppl

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
    Main function to generate text and compute sliding window perplexity.

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
    perplexities = []

    for prompt in all_prompts:
        ans, perplexity = compute_sliding_window_perplexity(generator if flag_llm == "llama" else llm, prompt, max_gen_len, temperature, top_p, flag_llm)
        answers.append(ans)
        perplexities.append(perplexity)

    # Save generated outputs
    with open(f"./Inputs&Outputs/{path}/outputs-{ckpt_dir}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(answers))

    # Save perplexities
    with open(f"./Inputs&Outputs/{path}/sliding_perplexities-{ckpt_dir}.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(perplexities))

    print("Text generation and sliding window perplexity computation complete!")

if __name__ == "__main__":
    fire.Fire(main)
