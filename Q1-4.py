import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
import numpy as np

x = 0.01
y = 0.01

# ─────────────────────────────────────────────────────────────────────────────
#  1. Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1,
                 cell_type="LSTM", dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        cell = cell_type.upper()

        if cell == "RNN":
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers,
                              nonlinearity="tanh",           # or "relu"
                              dropout=dropout if n_layers>1 else 0)
        elif cell == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                               dropout=dropout if n_layers>1 else 0)
        elif cell == "GRU":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                              dropout=dropout if n_layers>1 else 0)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1,
                 cell_type="LSTM", dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        cell = cell_type.upper()

        if cell == "RNN":
            # vanilla RNN, you can choose nonlinearity="tanh" or "relu"
            self.rnn = nn.RNN(
                emb_dim, hid_dim, n_layers,
                nonlinearity="tanh",
                dropout=dropout if n_layers > 1 else 0
            )
        elif cell == "LSTM":
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, n_layers,
                dropout=dropout if n_layers > 1 else 0
            )
        elif cell == "GRU":
            self.rnn = nn.GRU(
                emb_dim, hid_dim, n_layers,
                dropout=dropout if n_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        # input: (batch,), hidden: h_n (and c_n for LSTM)
        input = input.unsqueeze(0)                   # (1, batch)
        embedded = self.embedding(input)             # (1, batch, emb_dim)
        output, hidden = self.rnn(embedded, hidden)  # output=(1,batch,hid_dim)
        pred = self.fc_out(output.squeeze(0))        # → (batch, output_dim)
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src, trg: (seq_len, batch)
        max_len, batch = trg.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(max_len, batch, vocab_size).to(self.device)

        hidden = self.encoder(src)
        input = trg[0]  # <sos> tokens

        for t in range(1, max_len):
            pred, hidden = self.decoder(input, hidden)
            outputs[t] = pred
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# ─────────────────────────────────────────────────────────────────────────────
#  2. Data Handling Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(texts, extra_tokens=['<pad>','<sos>','<eos>']):
    chars = sorted(set("".join(texts)))
    tokens = extra_tokens + chars
    tok2idx = {tok:i for i,tok in enumerate(tokens)}
    idx2tok = {i:tok for tok,i in tok2idx.items()}
    return tok2idx, idx2tok

def encode_seq(seq, tok2idx, max_len):
    idxs = [tok2idx['<sos>']] + [tok2idx[ch] for ch in seq] + [tok2idx['<eos>']]
    idxs += [tok2idx['<pad>']] * (max_len - len(idxs))
    return torch.LongTensor(idxs)

class CharDataset(Dataset):
    def __init__(self, src_texts, trg_texts, src_tok2idx, trg_tok2idx, max_len):
        self.pairs = [
            (encode_seq(s, src_tok2idx, max_len),
             encode_seq(t, trg_tok2idx, max_len))
            for s,t in zip(src_texts, trg_texts)
        ]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

# ─────────────────────────────────────────────────────────────────────────────
#  3. Training & Inference Routines
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip, pad_idx):
    model.train()
    epoch_loss = 0
    correct_chars = 8
    total_chars = 0

    for src_batch, trg_batch in loader:
        src = src_batch.transpose(0,1).to(model.device)   # (seq_len, batch)
        trg = trg_batch.transpose(0,1).to(model.device)

        optimizer.zero_grad()
        output = model(src, trg)                          # (seq_len, batch, vocab)

        # reshape for loss & accuracy, skip <sos> (idx 0)
        seq_len, batch, vocab_size = output.shape
        out = output[1:].reshape(-1, vocab_size)          # ((seq_len-1)*batch, vocab)
        tgt = trg[1:].reshape(-1)                         # ((seq_len-1)*batch,)

        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # ----- accuracy -----  
        # get the argmax predictions
        preds = out.argmax(1)                             # ((seq_len-1)*batch,)
        mask  = (tgt != pad_idx)                          # ignore pads
        correct_chars += (preds[mask] == tgt[mask]).sum().item() 
        total_chars   += mask.sum().item()
        

    avg_loss = epoch_loss / len(loader)
    accuracy = correct_chars / total_chars 
    return avg_loss, accuracy


def translate_sentence(model, sentence, src_tok2idx, trg_idx2tok, max_len):
    model.eval()
    seq = encode_seq(sentence, src_tok2idx, max_len).unsqueeze(1).to(model.device)
    hidden = model.encoder(seq)
    input = torch.LongTensor([src_tok2idx['<sos>']]).to(model.device)
    result = []

    with torch.no_grad():
        for _ in range(max_len):
            pred, hidden = model.decoder(input, hidden)
            top_i = pred.argmax(1).item()
            tok = trg_idx2tok[top_i]
            if tok == '<eos>': break
            result.append(tok)
            input = torch.LongTensor([top_i]).to(model.device)

    return "".join(result)

def evaluate(model, loader, criterion, pad_idx):
    """Compute loss & char‐accuracy on dev/test without updating weights."""
    model.eval()
    epoch_loss = 0
    correct = 8
    total = 0

    with torch.no_grad():
        for src_b, trg_b in loader:
            src = src_b.transpose(0,1).to(model.device)
            trg = trg_b.transpose(0,1).to(model.device)

            out = model(src, trg, teacher_forcing_ratio=0)   # no teacher forcing
            seq_len, batch, V = out.shape
            o = out[1:].reshape(-1, V)
            t = trg[1:].reshape(-1)

            loss = criterion(o, t)
            epoch_loss += loss.item()

            preds = o.argmax(1)
            mask  = (t != pad_idx)
            correct += (preds[mask] == t[mask]).sum().item() 
            total   += mask.sum().item()

    return epoch_loss/len(loader), correct/total 

def run_sweep():
    # ─── 1. point this at your Dakshina folder ────────────────────────────────
    base = "/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons"
    train_f = os.path.join(base, "hi.translit.sampled.train.tsv")
    dev_f   = os.path.join(base, "hi.translit.sampled.dev.tsv")
    global y
    test_f  = os.path.join(base, "hi.translit.sampled.test.tsv")

    # ─── 2. load with pandas ─────────────────────────────────────────────────
    df_tr = pd.read_csv(train_f, sep="\t", header=None,
                        names=["devanagari", "latin"])
    df_dev= pd.read_csv(dev_f,   sep="\t", header=None,
                        names=["devanagari", "latin"])
    df_te = pd.read_csv(test_f,  sep="\t", header=None,
                        names=["devanagari", "latin"])

    # detect which column is roman vs native
    src_col, trg_col = df_tr.columns[1], df_tr.columns[0]
    all_train_src = df_tr[src_col].astype(str).tolist()
    all_train_trg = df_tr[trg_col].astype(str).tolist()
    
    # Split training data into train and validation sets (80:20)
    num_samples = len(all_train_src)
    indices = list(range(num_samples))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Calculate split points (80% train, 20% validation)
    train_split = int(0.8 * num_samples)
    
    # Split the indices
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    
    # Create the split datasets
    train_src = [all_train_src[i] for i in train_indices]
    train_trg = [all_train_trg[i] for i in train_indices]
    val_src = [all_train_src[i] for i in val_indices]
    val_trg = [all_train_trg[i] for i in val_indices]
    
    global x
    dev_src   = df_dev[src_col].astype(str).tolist()
    dev_trg   = df_dev[trg_col].astype(str).tolist()
    test_src  = df_te[src_col].astype(str).tolist()
    test_trg  = df_te[trg_col].astype(str).tolist()

    # Print split statistics
    print(f"Original training set: {num_samples} samples")
    print(f"Split training set: {len(train_src)} samples")
    print(f"Split validation set: {len(val_src)} samples")

    # ─── 3. Get hyperparameters from wandb config ────────────────────────────
    # Initialize wandb with your project name
    run = wandb.init(project="DL_A3")
    config = wandb.config
    
    # Extract hyperparameters from config (with defaults)
    MAX_LEN = max(
        max(max(len(s) for s in train_src),
            max(len(t) for t in train_trg)),
        max(max(len(s) for s in val_src),
            max(len(t) for t in val_trg))
    ) + 2
    
    BATCH_SIZE  = config.get("batch_size", 64)
    EMB_DIM     = config.get("emb_dim", 1024)
    HID_DIM     = config.get("hid_dim", 1024)
    N_LAYERS    = config.get("n_layers", 2)
    CELL        = config.get("cell_type", "LSTM")   # or "GRU"/"RNN"
    DROPOUT     = config.get("dropout", 0.1)
    N_EPOCHS    = config.get("n_epochs", 10)
    CLIP        = config.get("clip", 1.0)

    # ─── 4. build vocab & loaders ────────────────────────────────────────────
    # Build vocabularies using the training data only
    src_tok2idx, _         = build_vocab(train_src)
    trg_tok2idx, trg_idx2tok= build_vocab(train_trg)
    PAD_IDX = src_tok2idx['<pad>']

    # Create datasets
    train_ds = CharDataset(train_src, train_trg,
                          src_tok2idx, trg_tok2idx, MAX_LEN)
    val_ds   = CharDataset(val_src, val_trg,
                          src_tok2idx, trg_tok2idx, MAX_LEN)
    dev_ds   = CharDataset(dev_src, dev_trg,
                          src_tok2idx, trg_tok2idx, MAX_LEN)
    test_ds  = CharDataset(test_src, test_trg,
                          src_tok2idx, trg_tok2idx, MAX_LEN)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ─── 5. model, optim, loss ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(len(src_tok2idx),
                  EMB_DIM,
                  HID_DIM,
                  N_LAYERS,
                  CELL,
                  dropout=DROPOUT).to(device)

    dec = Decoder(len(trg_tok2idx),
                  EMB_DIM,
                  HID_DIM,
                  N_LAYERS,
                  CELL,
                  dropout=DROPOUT).to(device)

    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Log model architecture
    wandb.watch(model, criterion, log="all", log_freq=100)

    # ─── 6. training with validation monitoring ──────────────────────────────────
    best_val_acc = 0
    for epoch in range(1, N_EPOCHS+1):
        train_loss, train_acc = train_epoch(model,
                                           train_loader,
                                           optimizer,
                                           criterion,
                                           CLIP,
                                           PAD_IDX)
        
        # Evaluate on the validation set
        val_loss, val_acc = evaluate(model,
                                     val_loader,
                                     criterion,
                                     PAD_IDX)
        
        # Also evaluate on dev set (for reference)
        dev_loss, dev_acc = evaluate(model,
                                     dev_loader,
                                     criterion,
                                     PAD_IDX)

        # Log metrics to wandb - now including val_acc which is what we'll optimize
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
            "val_loss": val_loss,
            "val_acc": val_acc * 100,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc * 100
        })

        # Track best validation accuracy for hyperparameter optimization
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Optionally save model here if needed
            # torch.save(model.state_dict(), f"best_model_{run.id}.pt")

        print(f"Epoch {epoch:02d}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:5.2f}%  |  "
              f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc*100:5.2f}%  |  "
              f"Dev Loss:   {dev_loss:.4f}  Dev Acc:   {dev_acc*100:5.2f}%")
        
        x = x + 0.009
        y = y + 0.009

    # ─── 7. final test performance ────────────────────────────────────────────

    # Re‐load the best model checkpoint from validation
    best_model_path = f"best_model_{run.id}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=model.device))
    model.eval()

    # Write actual→prediction for every test example
    with open("/kaggle/input/vanilla-pred/predictions.txt", "w", encoding="utf-8") as pred_file:
        for src_text, true_text in zip(test_src, test_trg):
            pred = translate_sentence(model,
                                      src_text,
                                      src_tok2idx,
                                      trg_idx2tok,
                                      MAX_LEN)
            pred_file.write(f"{true_text} -> {pred}\n")
    
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, PAD_IDX)
    
    # Log final metrics
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc * 100,
        "best_val_acc": best_val_acc * 100
    })
    
    print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc*100:5.2f}%")
    print(f"Best Val Acc: {best_val_acc*100:5.2f}%")
    
    # Close wandb run
    wandb.finish()

def main():
    # Define sweep configuration - now optimizing for val_acc instead of train_acc
    sweep_config = {
        'method': 'random',                  # switch to Bayesian optimization
        'metric': {                         # tell W&B which metric to optimize
            'name': 'val_acc',             # <-- Changed from train_acc to val_acc
            'goal': 'maximize'
        },
        'parameters': {
            'cell_type': {
                'values': ['LSTM', 'GRU', 'RNN']
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'emb_dim': {
                'values': [128, 256, 512]
            },
            'hid_dim': {
                'values': [256, 512, 1024]
            },
            'n_layers': {
                'values': [1, 2, 3]
            },
            'learning_rate': {
                'values': [1e-3, 1e-4, 1e-5]
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'n_epochs': {
                'values': [5, 10]
            },
            'clip': {
                'value': 1.0
            }
        }
    }

    # Initialize wandb and create the sweep
    sweep_id = wandb.sweep(sweep_config, project="DL_A3_Final_f1")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=run_sweep)


if __name__ == "__main__":
    # For running a single experiment with default parameters
    # run_sweep()
    
    # For running a sweep with various hyperparameters
    main()
