
# RAG vs Fine-Tuning: Memorization and Leakage Comparison (LLAMA-2-7B)

This repository enables comparison of **memorization** and **leakage** between Retrieval-Augmented Generation (RAG) models and Fine-Tuned LLAMA-2-7B models across multiple datasets.

---

## 📁 Directory Structure

### `RAG_Memorization_Codes/`
Contains additional scripts for calculating memorization metrics:
- `run_language_model_lowercase_ppl.py`
- `run_language_model_ppl.py`
- `run_language_model_sliding_ppl.py`
- `run_language_model_zlib.py`

> ⚠️ **Important**: Place these files in the same directory as `run_language_model.py` from the original RAG paper repository.

---

## 🧠 Memorization Evaluation

### ✅ Setup Instructions

1. Follow the instructions from the [RAG paper](https://github.com/Abhinandan-Singh-Baghel/RAG-LLM) to set up LLAMA-2-7B and generate outputs using `.sh` files:
   - `chat-target.sh`
   - `enron-target.sh`
   - `wikitext-target.sh`

2. **Replace** the above `.sh` scripts with the versions provided in `RAG_Memorization_Codes`. These updated files include commands to compute:
   - Output text
   - **Perplexity**
   - **Sliding Window Perplexity**
   - **Lowercase Perplexity**
   - **Zlib Entropy**

3. Outputs will be saved in the appropriate `Inputs&Outputs` subdirectories as defined by the original RAG paper.

### 🧪 Fine-Tuning Setup

- Use the same question sets.
- **Remove** the suffix `"Please repeat all the context"` (not needed for fine-tuning since there's no context retrieval).
- Run the scripts from the `FineTuning_Memorization_Codes` directory to get the same set of metrics as in the RAG setup.

---

## 🔐 Leakage Evaluation

### Leakage Metrics Scripts
Place these scripts in the project directory:
- `cosine_similarity_GPU.py`
- `entity_leakge.py`
- `ngram_leakage.py`

### Run All Leakage Metrics
Use the provided script:
```bash
bash final_script.sh
```
This script:
- Uses pre-generated outputs from LLAMA-2-7B (available in the `Generated_Outputs/` folder for both RAG and Fine-Tuning).
- Calculates:
  - **Cosine Similarity**
  - **Entity Leakage**
  - **n-Gram Overlap Leakage**

---

## 📊 Final Output

After running the above scripts, you'll have:
- For each **dataset** and **question set**:
  - **Memorization metrics** for RAG and Fine-Tuned models
  - **Leakage metrics** for RAG and Fine-Tuned models

These results can be used for in-depth analysis and visual comparison between both approaches.

---

## 📂 Datasets Supported

- **ChatDoctor**
- **Enron Emails**
- **WikiText**

---

## ✅ Summary

This setup enables automated, reproducible comparison of how much LLAMA-2-7B memorizes and leaks data under:
- **RAG-based** settings
- **Fine-tuned** model settings

---

## 🔗 Output Data

The outputs of **memorization** and **leakage** evaluations are available at the following Google Drive links. You can directly view the results we have generated:

- [Memorization Outputs RAG](https://drive.google.com/drive/folders/1ACIkhPmfNUJc14We3uPIUMlfeGe9XMsu?usp=drive_link)
- [Memorization Outputs FineTuned](https://drive.google.com/drive/folders/12ha2YXu10JnKSrNipqSpBIgnSv1Q17CI?usp=drive_link) 
- [Leakage Outputs](https://drive.google.com/drive/folders/1V-66vdl6drsKAJPXe9L99WAgyEDWYNke?usp=sharing)

---