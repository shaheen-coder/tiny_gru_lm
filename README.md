# GruLM 🧠  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

*A minimal GRU-based Language Model from scratch in PyTorch*

GruLM is a **lightweight language model** built using a stacked **GRU** architecture with optional **weight tying**, trained using **next-token prediction**.  
This project is intentionally simple and educational — designed to help you understand **language modeling fundamentals without transformers**.

---

## ✨ Features

- 🔁 GRU-based causal language model
- 🔗 Optional **weight tying** (embedding ↔ output projection)
- 🧩 SentencePiece tokenizer support
- 🧪 Token-level accuracy + perplexity tracking
- 🎯 Top-K + temperature sampling for inference
- 🖥️ CPU-first, CUDA compatible

---

## 📂 Project Structure

```text
.
├── README.md
├── dataloader.py # Tokenizer + Padded dataset loading 
├── grulm.pt # output model
├── inference.py # Text Generation Script
├── model.py # GRU Language model 
├── tokenizer.model # SentencePiece tokenizer model
├── tokenizer.py # SenetencePiece custom tokenizer trainer
└── train.py # Training Loop
```
---

🧠 Model Architecture

Input IDs
   ↓
Embedding (V → E)
   ↓
GRU (E → H) × N layers
   ↓
Linear (H → V)
   ↓
Next-token logits

Key Details

Causal LM (predicts x[t+1] from x[:t])

Batch-first GRU

Weight tying supported if emb_dim == hdn_dim

No attention, no tricks — pure recurrent modeling



---

⚙️ Requirements

pip install torch sentencepiece

Tested with:

Python ≥ 3.12


---

🏋️ Training

The training loop uses teacher forcing with shifted inputs:

x = tokens[:, :-1]
y = tokens[:, 1:]

Run Training

python train.py

Training Highlights

CrossEntropyLoss with ignore_index = pad_id

Token-level accuracy (ignores padding)

Gradient clipping (clip_grad_norm_)

Perplexity reporting per epoch


Example output:

[Epoch 2] Sample 150/1000 | Loss 3.21 | TokenAcc 0.34
Avg NLL     : 3.05
Perplexity : 21.1


---

🗣️ Inference / Text Generation

Supports:

Temperature scaling

Top-K sampling

Autoregressive decoding


Run Inference

python inference.py

Example:

Enter your prompt: hello world
Output:
<prompt> hello world <ai> this is a simple grulm demo ...

Sampling Logic

logits = logits / temperature
top_k filtering
softmax → multinomial sampling


---

🙌 Acknowledgements

PyTorch

SentencePiece

Classic RNN / LM literature

---
