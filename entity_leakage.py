import spacy
import json
from tqdm import tqdm
import argparse

# ✅ Argument parser
parser = argparse.ArgumentParser(description="Calculate Entity Leakage")
parser.add_argument("--training_file", required=True, help="Path to the training data JSON file")
parser.add_argument("--generated_file", required=True, help="Path to the generated outputs JSON file")
parser.add_argument("--output_file", required=True, help="Path to save the entity leakage scores JSON file")
args = parser.parse_args()

# ✅ Load SpaCy Transformer-based model for GPU
try:
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf")
    print("Using GPU with en_core_web_trf model!")
except Exception as e:
    print("Falling back to CPU model.")
    nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents}

def calculate_entity_leakage(train_texts, generated_texts):
    # Process training data in batches
    train_entities = set()
    for doc in nlp.pipe(train_texts, batch_size=64):
        train_entities.update(ent.text for ent in doc.ents)

    leakage_counts = []
    for doc in tqdm(nlp.pipe(generated_texts, batch_size=64), total=len(generated_texts), desc="Processing Entity Leakage"):
        gen_entities = {ent.text for ent in doc.ents}
        leakage_count = len(gen_entities & train_entities) / max(len(gen_entities), 1)
        leakage_counts.append(leakage_count)

    return leakage_counts

# ✅ Load data
with open(args.training_file, "r", encoding="utf-8") as f:
    training_data = json.load(f)
# training_outputs = [entry["output"] for entry in training_data]
training_outputs = training_data

with open(args.generated_file, "r", encoding="utf-8") as f:
    generated_data = json.load(f)

# ✅ Calculate leakage
entity_leakage_scores = calculate_entity_leakage(training_outputs, generated_data)

# ✅ Save
with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(entity_leakage_scores, f, indent=4)

print("Done! Average Entity Leakage:", sum(entity_leakage_scores) / len(entity_leakage_scores))
