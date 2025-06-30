import argparse
import json
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Check if CUDA (GPU) is available  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model

def calculate_zlib_entropy(text):
    return len(zlib.compress(bytes(text, 'utf-8')))

def generate_responses_and_entropies(questions, model, tokenizer, device, max_length=256):
    responses = []
    zlib_entropies = []
    
    for question in tqdm(questions, desc="Generating responses and computing zlib entropies"):
        inputs = tokenizer(question, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        responses.append(generated_text)
        
        # Compute zlib entropy for the generated response
        entropy = calculate_zlib_entropy(generated_text)
        zlib_entropies.append(entropy)
    
    return responses, zlib_entropies

def main(args):
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer("abhinandan88/llama-enron-finetuned")
    model = load_model("abhinandan88/llama-enron-finetuned", device)
    print("Model loaded successfully!")

    # Load questions from JSON file
    with open(args.input_file, "r") as f:
        questions = json.load(f)  # Expecting a list of questions

    print(f"Generating responses and computing zlib entropies for {len(questions)} questions...")
    responses, zlib_entropies = generate_responses_and_entropies(questions, model, tokenizer, device)

    # # Save output responses
    # with open(args.output_file, "w") as f:
    #     json.dump(responses, f, indent=4)

    # Save zlib entropies
    entropy_data = {"questions": questions, "responses": responses, "zlib_entropies": zlib_entropies}
    with open(args.entropy_file, "w") as f:
        json.dump(entropy_data, f, indent=4)
    
    # print(f"Responses saved to {args.output_file}")
    print(f"Zlib entropies saved to {args.entropy_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions")
    # parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with responses")
    parser.add_argument('--entropy_file', type=str, required=True, help="Path to save the output JSON file with zlib entropies")
    args = parser.parse_args()
    main(args)

# Example usage:
# python3 finetuned_enron_zlib.py --input_file question_ppl_email.json --entropy_file Target_Email_zlib.json
