# LLM Fundamentals

A hands-on course that builds intuition for how large language models work — from NumPy fundamentals all the way to a retrieval-augmented question-answering pipeline. It starts with four Python prerequisite notebooks covering NumPy, Matplotlib, and PyTorch, then moves through five AI phases. Each notebook motivates a concept by showing where the *previous* approach falls short, then implements the fix in runnable code.

Everything is designed to run on a laptop (CPU or Apple Silicon / MPS). The notebooks build toward a capstone: an **AI Customer Support assistant** that summarizes tickets and answers questions grounded in a knowledge base.

## Python prerequisites (start here if you're new to AI tooling)

Four short notebooks that build the numerical and framework foundations every AI phase relies on. If you're already comfortable with NumPy, Matplotlib, and PyTorch you can skip straight to Phase 1.

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| PY 01 | `notebooks/py01_numpy.ipynb` | Why AI uses arrays instead of lists, shapes and dimensions, vectorized operations, broadcasting, matrix multiplication, softmax, and cosine similarity. |
| PY 02 | `notebooks/py02_matplotlib.ipynb` | Plotting training loss curves, token frequency bar charts, weight-distribution histograms, attention heatmaps, and 2D embedding scatter plots. |
| PY 03 | `notebooks/py03_pytorch.ipynb` | PyTorch tensors, automatic differentiation (autograd), building models with `nn.Module`, and the forward → loss → backward → update training loop. |
| PY 04 | `notebooks/py04_before_you_start_ai.ipynb` | Tool overview (PyTorch, HuggingFace Transformers & Datasets, ChromaDB), a one-sentence glossary of AI terms, and an optional reading list of key papers. |

## The five phases

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 1 | `notebooks/01_foundations.ipynb` | Why RNNs struggle (vanishing gradients, the encoder–decoder bottleneck), how attention and self-attention fix it, and how the Transformer assembles it all. |
| 1b | `notebooks/01b_tokenizers_embeddings.ipynb` | Subword tokenization (BPE / WordPiece / SentencePiece), token IDs, and how embeddings turn tokens into meaning. Static vs. contextual embeddings. |
| 2 | `notebooks/02_text_generation.ipynb` | How decoder-only models pick the next token: greedy, beam search, sampling, temperature, top-k, and top-p (nucleus) sampling. |
| 3 | `notebooks/03_prompt_engineering.ipynb` | Steering output without changing the model: zero/one/few-shot prompting, chain-of-thought, output constraints, and measuring quality with ROUGE. |
| 4 | `notebooks/04_fine_tuning.ipynb` | Adapting a model to a task: full fine-tuning vs. LoRA/PEFT, catastrophic forgetting, and the parameter/speed trade-offs. |
| 5 | `notebooks/05_rag.ipynb` | Retrieval-Augmented Generation: embeddings → vector database (ChromaDB) → retrieval → grounded generation, plus chunking and failure modes. |

Work through the prerequisites first, then the five phases in order — each phase assumes the one before it. Every notebook ends with `YOUR UNDERSTANDING` reflection prompts; filling these in is the point.

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
├── notebooks/
│   ├── py01_numpy.ipynb             # prerequisite: NumPy
│   ├── py02_matplotlib.ipynb        # prerequisite: Matplotlib
│   ├── py03_pytorch.ipynb           # prerequisite: PyTorch
│   ├── py04_before_you_start_ai.ipynb  # prerequisite: tools & glossary
│   ├── 01_foundations.ipynb         # phase 1: RNNs → Transformers
│   ├── 01b_tokenizers_embeddings.ipynb # phase 1b: tokenization & embeddings
│   ├── 02_text_generation.ipynb     # phase 2: decoding strategies
│   ├── 03_prompt_engineering.ipynb  # phase 3: prompt engineering
│   ├── 04_fine_tuning.ipynb         # phase 4: fine-tuning & LoRA
│   └── 05_rag.ipynb                 # phase 5: RAG pipeline
├── data/             # datasets used by the notebooks
├── binder/           # Binder environment configuration
├── requirements.txt  # pinned dependencies
├── .env.example      # template for your OpenAI key (optional)
└── outputs/          # fine-tuning checkpoints (created at runtime, git-ignored)
```

## License

This course is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — you are free to use, adapt, and share it with attribution.
