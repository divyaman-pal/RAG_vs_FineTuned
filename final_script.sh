#!/bin/bash

# N-gram runs
python3 ngram_leakage.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/Fine-Tuned/Chatdoctor/output.json --output_file ngram_finetuned_chatdoctor.json


python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt1/output.json --output_file ngram_finetuned_enron_qrt1.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt2/output.json --output_file ngram_finetuned_enron_qrt2.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt3/output.json --output_file ngram_finetuned_enron_qrt3.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt4/output.json --output_file ngram_finetuned_enron_qrt4.json



python3 ngram_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt1/output.json --output_file ngram_finetuned_wikitext_qrt1.json
python3 ngram_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt2/output.json --output_file ngram_finetuned_wikitext_qrt2.json


Now for RAG


python3 ngram_leakage.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/RAG/Chatdoctor/output.json --output_file ngram_RAG_chatdoctor.json


python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt1/output.json --output_file ngram_RAG_enron_qrt1.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt2/output.json --output_file ngram_RAG_enron_qrt2.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt3/output.json --output_file ngram_RAG_enron_qrt3.json
python3 ngram_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt4/output.json --output_file ngram_RAG_enron_qrt4.json



python3 ngram_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt1/output.json --output_file ngram_RAG_wikitext_qrt1.json
python3 ngram_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt2/output.json --output_file ngram_RAG_wikitext_qrt2.json





Entity leakage runs


python3 entity_leakage.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/Fine-Tuned/Chatdoctor/output.json --output_file entity_leakage_finetuned_chatdoctor.json


python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt1/output.json --output_file entity_leakage_finetuned_enron_qrt1.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt2/output.json --output_file entity_leakage_finetuned_enron_qrt2.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt3/output.json --output_file entity_leakage_finetuned_enron_qrt3.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt4/output.json --output_file entity_leakage_finetuned_enron_qrt4.json



python3 entity_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt1/output.json --output_file entity_leakage_finetuned_wikitext_qrt1.json
python3 entity_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt2/output.json --output_file entity_leakage_finetuned_wikitext_qrt2.json


# Now for RAG


python3 entity_leakage.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/RAG/Chatdoctor/output.json --output_file entity_leakage_RAG_chatdoctor.json


python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt1/output.json --output_file entity_leakage_RAG_enron_qrt1.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt2/output.json --output_file entity_leakage_RAG_enron_qrt2.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt3/output.json --output_file entity_leakage_RAG_enron_qrt3.json
python3 entity_leakage.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt4/output.json --output_file entity_leakage_RAG_enron_qrt4.json



python3 entity_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt1/output.json --output_file entity_leakage_RAG_wikitext_qrt1.json
python3 entity_leakage.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt2/output.json --output_file entity_leakage_RAG_wikitext_qrt2.json






Cosine Similarity



python3 cosine_similarity_GPU.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/Fine-Tuned/Chatdoctor/output.json --output_file cosine_similarity_finetuned_chatdoctor.json

python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron-Mail/qrt1/output.json --output_file cosine_similarity_finetuned_enron_qrt1.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt2/output.json --output_file cosine_similarity_finetuned_enron_qrt2.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt3/output.json --output_file cosine_similarity_finetuned_enron_qrt3.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/Fine-Tuned/Enron_Mail/qrt4/output.json --output_file cosine_similarity_finetuned_enron_qrt4.json



python3 cosine_similarity_GPU.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt1/output.json --output_file cosine_similarity_finetuned_wikitext_qrt1.json
python3 cosine_similarity_GPU.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/Fine-Tuned/Wikitext/qrt2/output.json --output_file cosine_similarity_finetuned_wikitext_qrt2.json


Now for RAG


python3 cosine_similarity_GPU.py --training_file formatted_chatdoctor_new.json --generated_file Generated_Outputs/RAG/Chatdoctor/output.json --output_file cosine_similarity_RAG_chatdoctor.json


python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt1/output.json --output_file cosine_similarity_RAG_enron_qrt1.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt2/output.json --output_file cosine_similarity_RAG_enron_qrt2.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt3/output.json --output_file cosine_similarity_RAG_enron_qrt3.json
python3 cosine_similarity_GPU.py --training_file formatted_enron_emails.json --generated_file Generated_Outputs/RAG/Enron_Mail/qrt4/output.json --output_file cosine_similarity_RAG_enron_qrt4.json



python3 cosine_similarity_GPU.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt1/output.json --output_file cosine_similarity_RAG_wikitext_qrt1.json
python3 cosine_similarity_GPU.py --training_file formatted_wikitext.json --generated_file Generated_Outputs/RAG/Wikitext/qrt2/output.json --output_file cosine_similarity_RAG_wikitext_qrt2.json

