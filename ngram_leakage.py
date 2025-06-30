
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

# ✅ Config
N = 3  # n-gram size
BATCH_SIZE = 100  # still used for pre-loading
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Fast n-gram extraction as sets
def text_to_ngram_set(text, n=N):
    tokens = tokenizer.encode(text.lower(), add_special_tokens=False)
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

# ✅ Overlap calculation using set intersection
def calculate_overlap_set(generated, training_sets):
    gen_ngrams = text_to_ngram_set(generated)
    if not gen_ngrams:
        return 0.0

    max_overlap = 0.0
    for train_ngrams in training_sets:
        if not train_ngrams:
            continue
        overlap = len(gen_ngrams & train_ngrams) / len(train_ngrams)
        max_overlap = max(max_overlap, overlap)
    return max_overlap

# ✅ Main script
parser = argparse.ArgumentParser()
parser.add_argument("--training_file", required=True)
parser.add_argument("--generated_file", required=True)
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

# ✅ Load training and generated data
with open(args.training_file, "r") as f:
    training_texts = json.load(f)
with open(args.generated_file, "r") as f:
    generated_texts = json.load(f)

# ✅ Precompute n-gram sets
print("Generating training n-gram sets...")
training_ngram_sets = [text_to_ngram_set(text) for text in tqdm(training_texts)]

# ✅ Process overlaps (faster!)
print("Computing overlaps...")
overlaps = [calculate_overlap_set(gen_text, training_ngram_sets) for gen_text in tqdm(generated_texts)]

# ✅ Save
with open(args.output_file, "w") as f:
    json.dump(overlaps, f, indent=2)

print("Done! Average Overlap:", sum(overlaps) / len(overlaps))
