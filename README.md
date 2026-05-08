# News Summarization using Fine-tuned BART

This repository contains a complete pipeline for fine-tuning a transformer-based model (**BART**) for the task of descriptive news summarization. The model was trained on the **ILSUM-1.0 English** dataset.

## Project Overview
The goal of this project is to take long-form news articles and generate concise, informative summaries using a sequence-to-sequence transformer architecture. We leverage the pre-trained `facebook/bart-base` model and fine-tune it to capture the nuances of news reporting.

## Dataset
- **Name:** ILSUM-1.0 English
- **Content:** News articles with corresponding headlines and summaries.
- **Filtering:** Articles were filtered to a maximum of $900$ tokens to ensure high-quality training without excessive truncation.

## Model Architecture
- **Base Model:** `facebook/bart-base`
- **Type:** Encoder-Decoder (Sequence-to-Seqence)
- **Parameters:** ~139 Million
- **Max Input Length:** 900 tokens
- **Max Output Length:** 128 tokens

## Fine-Tuning Details
- **Optimizer:** AdamW
- **Learning Rate:** $3 \times 10^{-5}$
- **Batch Size:** 8
- **Epochs:** 5
- **Mixed Precision:** FP16 enabled for faster training on CUDA.

## Evaluation Results

The model was evaluated on a held-out test set using **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) and **BERTScore**.

### Test Set Performance (ROUGE)
| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | $0.5011$ |
| **ROUGE-2** | $0.3764$ |
| **ROUGE-L** | $0.4522$ |
| **ROUGE-Lsum** | $0.4575$ |

### Semantic Evaluation (BERTScore)
| Metric | Score |
| :--- | :--- |
| **Precision** | $0.8489$ |
| **Recall** | $0.8369$ |
| **F1 Score** | $0.8425$ |

## Visualizations
The project includes several analysis components:
- **Training vs. Validation Loss:** Monitoring convergence during fine-tuning.
- **ROUGE Progress:** Tracking improvement in n-gram overlap across epochs.
- **BERTScore Distribution:** Analyzing the semantic similarity distribution across the test set.

## Installation
To run this project, you need the following dependencies:
```bash
pip install transformers[torch] datasets rouge_score bert_score evaluate
```

## Usage
### Inference Demo
```python
from transformers import BartForConditionalGeneration, BartTokenizer

model_path = "./bart-ilsum-finetuned"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

article = "Your long news article text here..."
inputs = tokenizer(article, return_tensors="pt", max_length=900, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=128)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
```

## Summary of Findings
- The model shows strong performance with a **ROUGE-1 score of ~0.50**, indicating high overlap with ground-truth summaries.
- **BERTScore F1 of 0.84** suggests the generated summaries maintain a high degree of semantic similarity to the references.
- Best performance was achieved at **Epoch 5**, showing consistent improvement during training.
