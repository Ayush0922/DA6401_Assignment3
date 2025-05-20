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
#  1. Model Definitions with Attention
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
                              dropout=dropout if n_layers>1 else 0,
                              bidirectional=True)            # Use bidirectional RNN for attention
        elif cell == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                              dropout=dropout if n_layers>1 else 0,
                              bidirectional=True)            # Use bidirectional LSTM for attention
        elif cell == "GRU":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                              dropout=dropout if n_layers>1 else 0,
                              bidirectional=True)            # Use bidirectional GRU for attention
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
        
        self.hid_dim = hid_dim
        self.cell_type = cell
        # Linear layer to reduce bidirectional hidden states to a single direction
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        # src: [src_len, batch]
        embedded = self.embedding(src)  # [src_len, batch, emb_dim]
        
        # outputs: [src_len, batch, hid_dim * 2] (bidirectional)
        # hidden: tuple of ([2, batch, hid_dim], [2, batch, hid_dim]) for LSTM
        # or just [2, batch, hid_dim] for GRU/RNN
        outputs, hidden = self.rnn(embedded)
        
        # Process hidden state for decoder
        if self.cell_type == "LSTM":
            # For LSTM, hidden is a tuple (h_n, c_n)
            h_n, c_n = hidden
            
            # Combine bidirectional hidden states
            h_n = self.fc(torch.cat((h_n[0], h_n[1]), dim=1)).unsqueeze(0)
            c_n = self.fc(torch.cat((c_n[0], c_n[1]), dim=1)).unsqueeze(0)
            
            # Return processed hidden state
            hidden = (h_n, c_n)
        else:
            # For GRU/RNN, hidden is just h_n
            # Combine bidirectional hidden states
            h_n = self.fc(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
            hidden = h_n
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        # Since encoder is bidirectional, its output dim is enc_hid_dim * 2
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, dec_hid_dim]
        # encoder_outputs: [src_len, batch, enc_hid_dim * 2]
        
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # Repeat the hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        # hidden: [src_len, batch, dec_hid_dim]
        
        # Calculate energy (alignment scores)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [src_len, batch, dec_hid_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [src_len, batch]
        
        # Softmax to get attention weights
        return torch.softmax(attention, dim=0)


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers=1,
                 cell_type="LSTM", dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        
        cell = cell_type.upper()
        self.cell_type = cell
        
        # Attention context vector and embedding will be concatenated as input to RNN
        if cell == "RNN":
            self.rnn = nn.RNN(emb_dim + (enc_hid_dim * 2), dec_hid_dim, n_layers,
                              nonlinearity="tanh",
                              dropout=dropout if n_layers > 1 else 0)
        elif cell == "LSTM":
            self.rnn = nn.LSTM(emb_dim + (enc_hid_dim * 2), dec_hid_dim, n_layers,
                               dropout=dropout if n_layers > 1 else 0)
        elif cell == "GRU":
            self.rnn = nn.GRU(emb_dim + (enc_hid_dim * 2), dec_hid_dim, n_layers,
                              dropout=dropout if n_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
        
        # Output layer: combines attention weighted context, decoder hidden state, and embedding
        self.fc_out = nn.Linear(emb_dim + (enc_hid_dim * 2) + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: tuple of ([1, batch, dec_hid_dim], [1, batch, dec_hid_dim]) for LSTM
        # or just [1, batch, dec_hid_dim] for GRU/RNN
        # encoder_outputs = [src_len, batch, enc_hid_dim * 2]
        
        input = input.unsqueeze(0)  # [1, batch]
        
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]
        
        # Get the correct hidden state for attention
        if self.cell_type == "LSTM":
            # For LSTM, hidden is a tuple (h_n, c_n)
            attn_hidden = hidden[0]
        else:
            # For GRU/RNN, hidden is just h_n
            attn_hidden = hidden
        
        # Calculate attention weights
        attn_weights = self.attention(attn_hidden, encoder_outputs)
        # attn_weights: [src_len, batch]
        
        # Create context vector by applying attention weights to encoder outputs
        attn_weights = attn_weights.unsqueeze(2)  # [src_len, batch, 1]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid_dim * 2]
        attn_weights = attn_weights.permute(1, 0, 2)  # [batch, src_len, 1]
        
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        # context: [batch, 1, enc_hid_dim * 2]
        
        context = context.permute(1, 0, 2)  # [1, batch, enc_hid_dim * 2]
        
        # Combine context vector with embedding as input to RNN
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [1, batch, emb_dim + enc_hid_dim * 2]
        
        # Get output and hidden state from RNN
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [1, batch, dec_hid_dim]
        # hidden: tuple of ([1, batch, dec_hid_dim], [1, batch, dec_hid_dim]) for LSTM
        # or just [1, batch, dec_hid_dim] for GRU/RNN
        
        # Concatenate embedded input, context, and RNN output for prediction
        embedded = embedded.squeeze(0)  # [batch, emb_dim]
        output = output.squeeze(0)  # [batch, dec_hid_dim]
        context = context.squeeze(0)  # [batch, enc_hid_dim * 2]
        
        # Predict next token
        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))
        # pred: [batch, output_dim]
        
        return pred, hidden


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src, trg: [seq_len, batch]
        max_len, batch = trg.shape
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch, vocab_size).to(self.device)
        
        # Get encoder outputs and hidden state
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to decoder is <sos> token
        input = trg[0]  # <sos> tokens
        
        for t in range(1, max_len):
            # Use current token, previous hidden state, and encoder outputs to predict next token
            pred, hidden = self.decoder(input, hidden, encoder_outputs)
            
            # Store prediction
            outputs[t] = pred
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = pred.argmax(1)
            
            # Next input is either ground truth or predicted token
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
    
    with torch.no_grad():
        # Get encoder outputs and hidden state
        encoder_outputs, hidden = model.encoder(seq)
        
        # First input is <sos> token
        input = torch.LongTensor([src_tok2idx['<sos>']]).to(model.device)
        
        result = []
        
        for _ in range(max_len):
            # Get predictions from decoder
            pred, hidden = model.decoder(input, hidden, encoder_outputs)
            
            # Get top predicted token
            top_i = pred.argmax(1).item()
            
            # Convert token to character
            tok = trg_idx2tok[top_i]
            
            # Stop if <eos> token
            if tok == '<eos>': 
                break
                
            result.append(tok)
            
            # Use predicted token as next input
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
    run = wandb.init(project="DL_A3_Attention")
    config = wandb.config
    
    # Extract hyperparameters from config (with defaults)
    MAX_LEN = max(
        max(max(len(s) for s in train_src),
            max(len(t) for t in train_trg)),
        max(max(len(s) for s in val_src),
            max(len(t) for t in val_trg))
    ) + 2
    
    BATCH_SIZE  = config.get("batch_size", 64)
    EMB_DIM     = config.get("emb_dim", 256)
    ENC_HID_DIM = config.get("enc_hid_dim", 256)  # Encoder hidden dim
    DEC_HID_DIM = config.get("dec_hid_dim", 256)  # Decoder hidden dim
    N_LAYERS    = config.get("n_layers", 1)       # Single layer as requested
    CELL        = config.get("cell_type", "LSTM") # or "GRU"/"RNN"
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
    
    # Create encoder with attention
    enc = Encoder(len(src_tok2idx),
                 EMB_DIM,
                 ENC_HID_DIM,
                 N_LAYERS,
                 CELL,
                 dropout=DROPOUT).to(device)

    # Create attention decoder
    dec = AttentionDecoder(len(trg_tok2idx),
                          EMB_DIM,
                          ENC_HID_DIM,
                          DEC_HID_DIM,
                          N_LAYERS,
                          CELL,
                          dropout=DROPOUT).to(device)

    # Create seq2seq model with attention
    model = Seq2SeqWithAttention(enc, dec, device).to(device)

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

        # Log metrics to wandb
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
            # Save the best model
            torch.save(model.state_dict(), f"best_model_attention_{run.id}.pt")

        print(f"Epoch {epoch:02d}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:5.2f}%  |  "
              f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc*100:5.2f}%  |  "
              f"Dev Loss:   {dev_loss:.4f}  Dev Acc:   {dev_acc*100:5.2f}%")
        
       

    # ─── 7. final test performance ────────────────────────────────────────────
    # Re‐load the best model checkpoint from validation
    best_model_path = f"best_model_attention_{run.id}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=model.device))
    model.eval()

    # Write actual→prediction for every test example
    with open(f"/kaggle/working/attention_predictions_{run.id}.txt", "w", encoding="utf-8") as pred_file:
        for src_text, true_text in zip(test_src, test_trg):
            pred = translate_sentence(model,
                                     src_text,
                                     src_tok2idx,
                                     trg_idx2tok,
                                     MAX_LEN)
            pred_file.write(f"{true_text} -> {pred}\n")
    
    # Evaluate on test set
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
    # Define sweep configuration for attention model
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'cell_type': {
                'values': ['LSTM', 'GRU']  # Removed RNN to simplify
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'emb_dim': {
                'values': [128, 256]
            },
            'enc_hid_dim': {
                'values': [128, 256]
            },
            'dec_hid_dim': {
                'values': [128, 256]
            },
            'n_layers': {
                'value': 1  # Fixed to 1 layer as requested
            },
            'learning_rate': {
                'values': [1e-3, 5e-4, 1e-4]
            },
            'batch_size': {
                'values': [32, 64]
            },
            'n_epochs': {
                'values': [10, 15]
            },
            'clip': {
                'value': 1.0
            }
        }
    }

    # Initialize wandb and create the sweep
    sweep_id = wandb.sweep(sweep_config, project="DL_A3_Attention")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=run_sweep)


if __name__ == "__main__":
    # For running a single experiment with default parameters
    #run_sweep()
    
    # For running a sweep with various hyperparameters
    main()
