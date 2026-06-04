# LLM Fundamentals

A hands-on, five-phase course that builds intuition for how large language models work — from the attention mechanism all the way to a retrieval-augmented question-answering pipeline. Each notebook motivates a concept by showing where the *previous* approach falls short, then implements the fix in runnable code.

Everything is designed to run on a laptop (CPU or Apple Silicon / MPS). The notebooks build toward a capstone: an **AI Customer Support assistant** that summarizes tickets and answers questions grounded in a knowledge base.

## The five phases

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 1 | `notebooks/01_foundations.ipynb` | Why RNNs struggle (vanishing gradients, the encoder–decoder bottleneck), how attention and self-attention fix it, and how the Transformer assembles it all. |
| 1b | `notebooks/01b_tokenizers_embeddings.ipynb` | Subword tokenization (BPE / WordPiece / SentencePiece), token IDs, and how embeddings turn tokens into meaning. Static vs. contextual embeddings. |
| 2 | `notebooks/02_text_generation.ipynb` | How decoder-only models pick the next token: greedy, beam search, sampling, temperature, top-k, and top-p (nucleus) sampling. |
| 3 | `notebooks/03_prompt_engineering.ipynb` | Steering output without changing the model: zero/one/few-shot prompting, chain-of-thought, output constraints, and measuring quality with ROUGE. |
| 4 | `notebooks/04_fine_tuning.ipynb` | Adapting a model to a task: full fine-tuning vs. LoRA/PEFT, catastrophic forgetting, and the parameter/speed trade-offs. |
| 5 | `notebooks/05_rag.ipynb` | Retrieval-Augmented Generation: embeddings → vector database (ChromaDB) → retrieval → grounded generation, plus chunking and failure modes. |

Work through them in order — each phase assumes the one before it. Every notebook ends with `YOUR UNDERSTANDING` reflection prompts; filling these in is the point.

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/samudassir/llm-fundamentals.git
cd llm-fundamentals

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### OpenAI API key (optional)

Notebooks 3 and 5 include optional sections that compare FLAN-T5 against `gpt-4o-mini`. These cells are skipped automatically if no key is present, so you can complete the course without one. To enable them:

```bash
cp .env.example .env
# then edit .env and paste your key
```

The notebooks load `../.env` relative to the `notebooks/` folder, so keep `.env` at the repo root.

## Hardware & runtime notes

- **Phases 1–3, 5** run comfortably on CPU in seconds to a couple of minutes per notebook.
- **Phase 4 (fine-tuning)** is the heaviest. It uses FLAN-T5-small on ~200 examples to stay laptop-friendly; expect roughly 5–10 minutes for full fine-tuning on CPU, less on MPS. LoRA is several times faster.
- Models and datasets download from the Hugging Face Hub on first run and are cached locally thereafter.
- The first run of each notebook pulls model weights (GPT-2 ~500 MB, FLAN-T5-base ~1 GB), so an initial internet connection is required.

## Project layout

```
llm-fundamentals/
├── notebooks/        # the five-phase course (run in order)
├── requirements.txt  # pinned dependencies
├── .env.example      # template for your OpenAI key (optional)
└── outputs/          # fine-tuning checkpoints (created at runtime, git-ignored)
```

## License

Add a license of your choice (e.g. MIT) if you intend others to reuse this material.
