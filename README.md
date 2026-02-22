# Small Language Model (SLM) on TinyStories

This project involves the implementation and training of a **custom transformer-based language model** designed to explore the core principles of the **attention mechanism** and text generation. While the original target was 50–60 million parameters, the final architecture consists of approximately **30 million parameters**.

---

## 📖 Overview

The model is trained on the **TinyStories dataset**, a collection of short stories that use a limited vocabulary suitable for young children. This makes it an ideal dataset for training smaller models to achieve high levels of linguistic coherence without requiring massive compute resources.

### Key Features

* **Custom Transformer Architecture**: Built from scratch using PyTorch.
* **GPT-2 Tokenization**: Utilizes the `tiktoken` library for efficient byte-pair encoding.
* **Weight Tying**: The embedding layer and language modeling head share weights to reduce parameter count.
* **Optimization Techniques**: Includes AdamW optimizer, linear warmup, and cosine annealing learning rate schedules.

---

## 🏗️ Model Architecture

The model follows a decoder-only transformer design. Below are the specific configurations used in this project:

| Component            | Specification       |
| -------------------- | ------------------- |
| **Layers (Blocks)**   | 6                   |
| **Attention Heads**   | 6                   |
| **Embedding Dimension** | 384                |
| **Block Size (Context)** | 128 tokens        |
| **Vocabulary Size**   | 50,257              |
| **Dropout Rate**      | 0.1                 |
| **Parameter Count**   | ~30 Million         |

---

## 🚀 Training Configuration

The model was trained using the following hyperparameters and environment settings:

* **Optimizer**: AdamW with linear warmup and cosine annealing.
* **Learning Rate**: Started at 1e-4 with a linear warmup of 1,000 steps, followed by cosine annealing to a minimum of 5e-4.
* **Batch Size**: 32 with a gradient accumulation of 32 steps.
* **Hardware**: Trained on a Tesla T4 GPU (via Google Colab).
* **Precision**: Mixed-precision training using `torch.amp.autocast` (float16).

---

## 📊 Training Results

The model demonstrated significant learning over 10,000 iterations.

### Loss Progress

* **Epoch 500**: Train Loss 9.4424 | Val Loss 9.4479.
* **Epoch 5000**: Train Loss 3.9900 | Val Loss 3.9966.
* **Epoch 9500**: Train Loss 2.9564 | Val Loss 2.9633.

> **Note**: The training and validation loss stayed closely aligned throughout, indicating the model did not significantly overfit the data.

---

## 🖋️ Text Generation Examples

The model uses **top-k sampling** to generate text. Here is an example of the model's output:

**Prompt**: *"Once upon a time there was a girl."*  
**Generated Text**:

> "...She loved to have sand and swimming. One day, she went along the lake and noticed a small rabbit came running. He was looking very high and it started to cry... He quickly helped him granted the superhero, tiara. Alex was so happy that he sang out."

---

## 🛠️ Requirements

To run the code in this repository, you need:

* `torch`
* `datasets`
* `tiktoken`
* `transformers` (for GPT-2 Tokenizer during inference)
* `matplotlib`
* `tqdm`
