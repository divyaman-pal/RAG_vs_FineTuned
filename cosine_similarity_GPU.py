import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Calculate cosine similarity between training and generated texts")
parser.add_argument("--training_file", type=str, required=True, help="Path to training data JSON file")
parser.add_argument("--generated_file", type=str, required=True, help="Path to generated data JSON file")
parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON file")
# parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing embeddings")
# parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="Model name for sentence embedding")
args = parser.parse_args()


batch_size = 8
model_name = "BAAI/bge-large-en-v1.5"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

class TextDataset(Dataset):
    """Custom dataset for loading text in batches"""
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_sentence_embedding(batch_texts):
    """Generate sentence embedding for a batch of texts"""
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def process_texts_with_dataloader(texts, batch_size):
    """Efficient text processing using DataLoader"""
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for batch in tqdm(dataloader, desc="Processing Embeddings", unit="batch"):
        emb = get_sentence_embedding(batch)
        embeddings.append(emb)
        torch.cuda.empty_cache()  # Free up GPU memory after each batch
    return np.vstack(embeddings)

def calculate_embedding_similarity(train_texts, generated_texts, batch_size):
    """Compute embedding similarity with optimized memory usage"""
    similarities = []

    # Precompute train embeddings
    train_embeddings = process_texts_with_dataloader(train_texts, batch_size=batch_size)

    # Process generated texts
    dataset = TextDataset(generated_texts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(dataloader, desc="Computing Similarity", unit="text"):
        gen_embedding = get_sentence_embedding(batch)
        sim_scores = cosine_similarity(gen_embedding, train_embeddings)[0]
        max_sim = float(max(sim_scores))  # Convert to Python float
        similarities.append(max_sim)
        torch.cuda.empty_cache()

    return similarities

def main():
    # Load training data
    with open(args.training_file, "r", encoding="utf-8") as f:
        training_texts = json.load(f)

    # Load generated data
    with open(args.generated_file, "r", encoding="utf-8") as f:
        generated_texts = json.load(f)

    # Compute embedding similarity
    embedding_similarities = calculate_embedding_similarity(training_texts, generated_texts, batch_size)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(embedding_similarities, f, indent=4)

    print("Embedding Similarity Scores:", embedding_similarities)
    print("Average Similarity:", np.mean(embedding_similarities))
    print(f"Results successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
