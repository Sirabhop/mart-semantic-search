# Semantic Search Engine

This repository is designed to provide end-to-end development of LLM models for semantic search engines.

**Features:**

- **Semantic understanding:** Leverages Large Language Models (LLMs) to understand the meaning behind search queries and documents.
- **End-to-end solution:**  Covers model selection, fine-tuning, deployment, and batch processing for scalable search.
- **Optimized for Databricks:** Includes specific instructions and code for deploying on the Databricks platform.

## **Repository Structure**
```
mart-semantic-search
├── models
├── src
└── deployment
```

## Getting Started

1. **Prerequisites:**
   - Databricks workspace (or adapt for your preferred platform)
   - Familiarity with sentence transformers and LLM concepts
   - Necessary dependencies (see `requirements.txt`)

2. **Installation:**
   ```bash
   git clone [invalid URL removed]
   cd mart-semantic-search
   pip install -r requirements.txt
   ```
3. **Model Selection and Fine-tuning:**
  - Explore and select the best sentence transformer model in models from 20+ models available on HuggingFace.

| file_label | validation_language | Accuracy@10 | Precision@10 | Recall@10 | MRR@10 | NDCG@10 | MAP@100 |
|------------|---------------------|-------------|--------------|-----------|--------|---------|---------|
| rbh-mart-miniLM-L12 | TH+EN | 82.22% | 25.27% | 13.52% | 59.55% | 28.98% | 16.33% |
| all-MiniLM-L6-v2 | TH+EN | 10.89% | 2.73% | 1.48% | 6.34% | 3.03% | 1.60% |
| LaBSE | TH+EN | 62.52% | 12.35% | 6.19% | 49.29% | 17.02% | 5.65% |
| paraphrase-multilingual-MiniLM-L12-v2 | TH+EN | 61.64% | 12.40% | 6.29% | 49.19% | 17.08% | 6.04% |
| jina-embeddings-v2-small-en | TH+EN | 9.12% | 2.02% | 1.03% | 5.84% | 2.43% | 0.89% |
| paraphrase-multilingual-mpnet-base-v2 | TH+EN | 62.26% | 12.48% | 6.22% | 49.77% | 17.25% | 5.95% |
| distiluse-base-multilingual-cased-v2 | TH+EN | 51.94% | 9.07% | 4.52% | 45.38% | 13.90% | 4.01% |
| distilbert-multilingual-nli-stsb-quora-ranking | TH+EN | 58.42% | 11.16% | 5.59% | 48.20% | 16.06% | 5.38% |
| use-cmlm-multilingual | TH+EN | 59.62% | 11.80% | 5.87% | 47.75% | 16.45% | 5.27% |
| stsb-xlm-r-multilingual | TH+EN | 59.05% | 11.72% | 5.76% | 47.96% | 16.43% | 5.24% |
| quora-distilbert-multilingual | TH+EN | 58.42% | 11.16% | 5.59% | 48.20% | 16.06% | 5.38% |
| paraphrase-xlm-r-multilingual-v1 | TH+EN | 57.75% | 10.68% | 5.34% | 47.36% | 15.53% | 4.85% |
| msmarco-MiniLM-L-6-v3 | TH+EN | 9.43% | 2.21% | 1.31% | 5.72% | 2.53% | 1.50% |
| msmarco-MiniLM-L-12-v3 | TH+EN | 9.18% | 2.28% | 1.35% | 5.79% | 2.61% | 1.54% |
| msmarco-distilbert-base-v4 | TH+EN | 10.94% | 2.80% | 1.61% | 6.43% | 3.10% | 1.74% |
| msmarco-roberta-base-v3 | TH+EN | 55.78% | 9.96% | 5.14% | 47.01% | 14.80% | 4.89% |
| msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned | TH+EN | 67.08% | 14.98% | 7.46% | 52.29% | 19.66% | 6.80% |
| finetuned-miniLM-L12 | EN | 78.38% | 22.16% | 16.50% | 46.16% | 24.08% | 18.05% |
| all-MiniLM-L6-v2 | EN | 86.49% | 20.54% | 16.16% | 36.90% | 20.68% | 18.11% |
| LaBSE | EN | 29.73% | 5.68% | 4.60% | 15.74% | 6.78% | 3.72% |
| paraphrase-multilingual-MiniLM-L12-v2 | EN | 27.03% | 5.41% | 4.33% | 16.89% | 6.67% | 4.00% |
| jina-embeddings-v2-small-en | EN | 51.35% | 11.08% | 8.56% | 31.64% | 13.09% | 5.85% |
| paraphrase-multilingual-mpnet-base-v2 | EN | 40.54% | 9.19% | 7.17% | 22.03% | 10.27% | 6.63% |
| distiluse-base-multilingual-cased-v2 | EN | 13.51% | 2.70% | 2.12% | 10.81% | 3.72% | 1.67% |
| distilbert-multilingual-nli-stsb-quora-ranking | EN | 18.92% | 4.32% | 3.20% | 12.70% | 5.19% | 3.15% |
| use-cmlm-multilingual | EN | 16.22% | 2.97% | 2.48% | 11.26% | 4.02% | 1.77% |
| stsb-xlm-r-multilingual | EN | 21.62% | 3.24% | 2.57% | 14.19% | 4.74% | 2.37% |
| quora-distilbert-multilingual | EN | 18.92% | 4.32% | 3.20% | 12.70% | 5.19% | 3.15% |
| paraphrase-xlm-r-multilingual-v1 | EN | 18.92% | 4.32% | 3.51% | 11.65% | 5.22% | 2.67% |
| msmarco-MiniLM-L-6-v3 | EN | 86.49% | 22.43% | 17.58% | 43.39% | 23.35% | 21.82% |
| msmarco-MiniLM-L-12-v3 | EN | 75.68% | 21.35% | 16.40% | 41.98% | 22.64% | 21.26% |
| msmarco-distilbert-base-v4 | EN | 91.89% | 25.68% | 20.08% | 47.09% | 26.40% | 22.44% |
| msmarco-roberta-base-v3 | EN | 86.49% | 23.51% | 18.62% | 46.00% | 25.03% | 18.74% |
| msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned | EN | 54.05% | 13.51% | 10.38% | 25.75% | 14.04% | 8.87% |

  - The winning model is `sirabhop/mart-multilingual-semantic-search-miniLM-L12`

4. **Deployment:**
  - Follow the instructions in deployment to set up your Databricks environment.
  - Deploy the fine-tuned model as a service endpoint.
  - (Optional) Configure batch processing for SQS (Simple Queue Service).
