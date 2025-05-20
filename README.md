# Character-level Sequence-to-Sequence Models for Transliteration

This repository contains Python code for character-level sequence-to-sequence (Seq2Seq) models, designed for transliteration tasks, specifically between English (Latin script) and Hindi (Devanagari script). It explores both a basic Seq2Seq architecture and an enhanced version incorporating attention mechanisms. The project leverages `wandb` for hyperparameter tuning and experiment tracking.

## Table of Contents

1.  [Q1-4.py: Vanilla Seq2Seq Model](https://www.google.com/search?q=%23q1-4py-vanilla-seq2seq-model)
      * [Overview](https://www.google.com/search?q=%23overview)
      * [Model Architecture](https://www.google.com/search?q=%23model-architecture)
      * [Hyperparameter Tuning with Weights & Biases](https://www.google.com/search?q=%23hyperparameter-tuning-with-weights--biases)
      * [Running the Vanilla Seq2Seq Model](https://www.google.com/search?q=%23running-the-vanilla-seq2seq-model)
2.  [Q5.py: Seq2Seq Model with Attention](https://www.google.com/search?q=%23q5py-seq2seq-model-with-attention)
      * [Overview](https://www.google.com/search?q=%23overview-1)
      * [Model Architecture](https://www.google.com/search?q=%23model-architecture-1)
      * [Hyperparameter Tuning with Weights & Biases](https://www.google.com/search?q=%23hyperparameter-tuning-with-weights--biases-1)
      * [Running the Attention-based Seq2Seq Model](https://www.google.com/search?q=%23running-the-attention-based-seq2seq-model)
3.  [q6.py: Small-Scale Attention Demo](https://www.google.com/search?q=%23q6py-small-scale-attention-demo)
      * [Overview](https://www.google.com/search?q=%23overview-2)
      * [Model Architecture](https://www.google.com/search?q=%23model-architecture-2)
      * [Running the Small-Scale Attention Demo](https://www.google.com/search?q=%23running-the-small-scale-attention-demo)
4.  [Installation](https://www.google.com/search?q=%23installation)
5.  [Usage](https://www.google.com/search?q=%23usage)

-----

## Q1-4.py: Vanilla Seq2Seq Model

### Overview

This script implements a basic character-level sequence-to-sequence model for transliteration. It consists of an Encoder-Decoder architecture using recurrent neural networks (RNNs), LSTMs, or GRUs. The model is trained to convert English words to their Hindi transliterations based on the Dakshina dataset. Weights & Biases (W\&B) is integrated for tracking experiments and performing hyperparameter sweeps.

### Model Architecture

The `Q1-4.py` script defines the following core components:

  * **Encoder**: An `nn.Module` that takes a sequence of input characters, embeds them, and processes them through an RNN (RNN, LSTM, or GRU) to produce a fixed-size context vector (the final hidden state).
  * **Decoder**: An `nn.Module` that takes the context vector from the encoder and sequentially generates output characters. It uses an embedding layer and an RNN (RNN, LSTM, or GRU) to predict the next character in the sequence. Teacher forcing can be applied during training.
  * **Seq2Seq**: An `nn.Module` that combines the encoder and decoder to form the complete transliteration model. It handles the forward pass, including teacher forcing for training.

### Hyperparameter Tuning with Weights & Biases

The script uses W\&B for hyperparameter optimization through sweeps. The `sweep_config` dictionary in `main()` defines the search space for various hyperparameters, including:

  * `cell_type`: `LSTM`, `GRU`, `RNN`
  * `dropout`: `0.1`, `0.2`, `0.3`
  * `emb_dim`: `128`, `256`, `512`
  * `hid_dim`: `256`, `512`, `1024`
  * `n_layers`: `1`, `2`, `3`
  * `learning_rate`: `1e-3`, `1e-4`, `1e-5`
  * `batch_size`: `32`, `64`, `128`
  * `n_epochs`: `5`, `10`
  * `clip`: `1.0` (fixed)

The sweep is configured to maximize `val_acc` (validation accuracy).

### Running the Vanilla Seq2Seq Model

To run a hyperparameter sweep for the vanilla Seq2Seq model:

1.  Ensure you have `wandb` installed and configured.
2.  Navigate to the directory containing `Q1-4.py`.
3.  Execute the script:
    ```bash
    python Q1-4.py
    ```
    This will initiate a W\&B sweep. Follow the instructions provided by W\&B in your terminal to start the sweep agent.

-----

## Q5.py: Seq2Seq Model with Attention

### Overview

This script enhances the basic Seq2Seq model by incorporating an attention mechanism. This allows the decoder to "pay attention" to different parts of the input sequence when generating each output character, which can significantly improve performance for sequence-to-sequence tasks. Like `Q1-4.py`, it uses the Dakshina dataset and integrates with W\&B for hyperparameter tuning.

### Model Architecture

The `Q5.py` script introduces modified and new components for attention:

  * **Encoder**: Similar to the vanilla encoder, but the RNN is **bidirectional** to capture context from both directions of the input sequence. It also includes a linear layer to transform the bidirectional hidden states into a single-direction hidden state suitable for the decoder.
  * **Attention**: A new `nn.Module` that calculates alignment scores between the decoder's hidden state and the encoder's outputs. It uses a linear layer and a `softmax` function to produce attention weights.
  * **AttentionDecoder**: A modified decoder that uses the `Attention` mechanism. The input to its RNN is a concatenation of the embedded input character and a context vector (a weighted sum of encoder outputs based on attention weights). The final prediction layer also considers the embedded input, RNN output, and context vector.
  * **Seq2SeqWithAttention**: Combines the bidirectional encoder and attention decoder to manage the forward pass and teacher forcing.

### Hyperparameter Tuning with Weights & Biases

The `sweep_config` in `Q5.py` is tailored for the attention model:

  * `cell_type`: `LSTM`, `GRU` (RNN is removed for simplicity based on expected performance)
  * `dropout`: `0.1`, `0.2`, `0.3`
  * `emb_dim`: `128`, `256`
  * `enc_hid_dim`: `128`, `256`
  * `dec_hid_dim`: `128`, `256`
  * `n_layers`: `1` (fixed as requested)
  * `learning_rate`: `1e-3`, `5e-4`, `1e-4`
  * `batch_size`: `32`, `64`
  * `n_epochs`: `10`, `15`
  * `clip`: `1.0` (fixed)

The goal remains to maximize `val_acc`.

### Running the Attention-based Seq2Seq Model

To run a hyperparameter sweep for the attention-based Seq2Seq model:

1.  Ensure you have `wandb` installed and configured.
2.  Navigate to the directory containing `Q5.py`.
3.  Execute the script:
    ```bash
    python Q5.py
    ```
    This will initiate a W\&B sweep. Follow the instructions provided by W\&B in your terminal to start the sweep agent.

-----

## q6.py: Small-Scale Attention Demo

### Overview

This script provides a simplified, self-contained demonstration of a Seq2Seq model with an attention mechanism on a very small, custom dataset. It's designed to illustrate the attention mechanism and its effect on character alignment during transliteration without the overhead of a large dataset or W\&B integration.

### Model Architecture

The `q6.py` script defines a minimal Seq2Seq with attention:

  * **Encoder**: A simple GRU-based encoder.
  * **Attention**: A standard attention mechanism that calculates attention weights between the decoder's hidden state and encoder outputs.
  * **Decoder**: A GRU-based decoder that uses the attention mechanism to create a context vector, which is then concatenated with the input embedding before being passed to the RNN. The output layer combines the decoder's hidden state and the context vector.

The script manually trains the model on a few English-Hindi word pairs and then demonstrates transliteration for a test word, printing how attention is distributed across the input characters for each generated output character.

### Running the Small-Scale Attention Demo

To run this demonstration:

1.  Navigate to the directory containing `q6.py`.
2.  Execute the script:
    ```bash
    python q6.py
    ```
    The output will show the transliteration of a random test word and highlight which input characters the model "paid attention" to for each output character.

-----

## Installation

### Prerequisites

  * Python 3.7+
  * pip

### Steps

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**

      * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
      * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**

    ```bash
    pip install torch pandas numpy wandb
    ```

4.  **Weights & Biases (wandb) Setup:**
    If you plan to run `Q1-4.py` or `Q5.py` with hyperparameter sweeps, you'll need to log in to your Weights & Biases account.

    ```bash
    wandb login
    ```

    Follow the prompts to enter your API key.

## Usage

### Data Preparation

The scripts expect the Dakshina dataset to be available at a specific path (`/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons`). Before running the main training scripts (`Q1-4.py` and `Q5.py`), ensure you have downloaded and placed the `hi.translit.sampled.train.tsv`, `hi.translit.sampled.dev.tsv`, and `hi.translit.sampled.test.tsv` files in this directory structure relative to where you run the script, or modify the `base` path in `run_sweep()` function within each script.

### Running the Models

  * **Vanilla Seq2Seq (Q1-4.py):**
    To start a W\&B sweep for hyperparameter optimization:

    ```bash
    python Q1-4.py
    ```

    This will print a command to start the W\&B agent (e.g., `wandb agent <sweep_id>`). Run this command in a new terminal to begin the sweep.

  * **Attention Seq2Seq (Q5.py):**
    To start a W\&B sweep for hyperparameter optimization:

    ```bash
    python Q5.py
    ```

    Similar to `Q1-4.py`, you'll get a command to start the W\&B agent.

  * **Small-Scale Attention Demo (q6.py):**
    To run the quick demonstration of attention:

    ```bash
    python q6.py
    ```

    This script will execute directly and print its output to the console.

Remember to activate your virtual environment before running any of the scripts.
