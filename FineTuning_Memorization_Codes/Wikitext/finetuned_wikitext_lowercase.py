import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

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

def calculate_lowercase_perplexity(model, tokenizer, text, device):
    lowercase_text = text.lower()
    inputs = tokenizer(lowercase_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    
    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()
    return perplexity

def generate_responses_and_perplexities(questions, model, tokenizer, device, max_length=1024):
    responses = []
    perplexities = []
    
    for question in tqdm(questions, desc="Generating responses and computing perplexities"):
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
        
        # Compute lowercase perplexity
        perplexity = calculate_lowercase_perplexity(model, tokenizer, generated_text, device)
        perplexities.append(perplexity)
    
    return responses, perplexities

def main(args):
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer("AG1010/llama2-wikitext")
    model = load_model("AG1010/llama2-wikitext", device)
    print("Model loaded successfully!")

    with open(args.input_file, "r") as f:
        questions = json.load(f)

    print(f"Generating responses and perplexities for {len(questions)} questions...")
    responses, perplexities = generate_responses_and_perplexities(questions, model, tokenizer, device)

    with open(args.output_file, "w") as f:
        json.dump(responses, f, indent=4)

    perplexity_data = {"questions": questions, "responses": responses, "perplexities": perplexities}
    with open(args.perplexity_file, "w") as f:
        json.dump(perplexity_data, f, indent=4)
    
    print(f"Responses saved to {args.output_file}")
    print(f"Perplexities saved to {args.perplexity_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with responses")
    parser.add_argument('--perplexity_file', type=str, required=True, help="Path to save the output JSON file with perplexities")
    args = parser.parse_args()
    main(args)
