import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle
import argparse
import mmap

# Argument parser setup
def setup_parser():
    parser = argparse.ArgumentParser(description="Model training for a Shakespearean language model")
    parser.add_argument('-b_s', type=str, required=True, help="Define batch size")
    parser.add_argument('--test', action='store_true', help="Test the model without training")
    parser.add_argument('--train_continue', action='store_true', help="Continue training the model")
    return parser

args = setup_parser().parse_args()

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device} device")

batch_size = int(args.batch_size)
block_size = 128
max_iters = 30000
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

# Reading Shakespeare text
with open('data/shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    

unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
print(f"Vocabulary Size: {vocab_size}")

# Character encoding and decoding
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

encode_text = lambda s: [char_to_index[char] for char in s]
decode_text = lambda l: ''.join([index_to_char[idx] for idx in l])

# Function to get a random text chunk from the Shakespeare file
def load_chunk(split):
    filename = 'data/shakespeare.txt'
    with open(filename, 'rb') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as memmap:
            file_size = len(memmap)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            memmap.seek(start_pos)
            block = memmap.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode_text(decoded_block), dtype=torch.long)
    return data

# Function to get a batch of data
def get_data_batch(split):
    data = load_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_batch = torch.stack([data[i:i + block_size] for i in ix])
    y_batch = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x_batch.to(device), y_batch.to(device)

# Evaluation loss estimation
@torch.no_grad()
def evaluate_loss():
    model.eval()
    loss_dict = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_data_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        loss_dict[split] = losses.mean()
    model.train()
    return loss_dict

# Transformer Head definition
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        attention_weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        attention_weights = attention_weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        value = self.value(x)
        return attention_weights @ value

# Multi-head Attention
class MultiHeadAttentionModule(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.projection(torch.cat([head(x) for head in self.heads], dim=-1)))

# Feed-forward network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttentionModule(n_head, head_size)
        self.feed_forward = PositionwiseFeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        return self.layer_norm2(x + feed_forward_output)

# Language Model Definition
class ShakespeareLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.position_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size)
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_idx, targets=None):
        B, T = input_idx.shape
        token_emb = self.token_embed(input_idx)
        pos_emb = self.position_embed(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def generate(self, input_idx, max_new_tokens, temperature = 0.9):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_idx)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            input_idx = torch.cat((input_idx, next_token), dim=1)
        return input_idx

# Initialize or load model
if not args.test and not args.train_continue:
    # Initialize model
    model = ShakespeareLanguageModel(vocab_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Track training losses
    training_losses = []

    # Training loop
    for step in range(max_iters):
        if step % eval_iters == 0:
            losses = evaluate_loss()
            training_losses.append((step, losses))
            print(f"Step: {step}, Losses: {losses}")

        x_batch, y_batch = get_data_batch('train')
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final Loss: {loss.item()}")

    # Save the model
    with open('shakespeare_model_5.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Training complete, model saved.")

    # Output training losses over time
    print("\nTraining Losses:")
    for step, losses in training_losses:
        print(f"Step {step}: Train Loss = {losses['train']}, Validation Loss = {losses['val']}")

elif args.train_continue:
    # Load the pre-trained model and continue training
    with open('shakespeare_model_4.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Track training losses
    training_losses = []

    # Continue training loop
    start_iter = 0  # You can adjust this if you saved a checkpoint
    for step in range(start_iter, max_iters):
        if step % eval_iters == 0:
            losses = evaluate_loss()
            print(f"Step: {step}, Losses: {losses}")
            training_losses.append((step, losses))

        x_batch, y_batch = get_data_batch('train')
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the model after continuing training
    with open('shakespeare_model_4.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Training complete, model saved.")

    # Output training losses over time
    print("\nTraining Losses:")
    for step, losses in training_losses:
        print(f"Step {step}: Train Loss = {losses['train']}, Validation Loss = {losses['val']}")
if args.test:
    with open('shakespeare_model_4.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    model.to(device)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_chars = decode_text(model.generate(context, max_new_tokens=100)[0].tolist())
    
    with open("generated_text.txt", "w", encoding="utf-8") as output_file:
        output_file.write(generated_chars)
    
    print("Generated text saved to generated_text.txt")
else:
    # Load the pre-trained model
    with open('shakespeare_model_4.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    model.to(device)

    # Generate text without training
    while True:
        prompt = input("Enter a prompt: ")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_chars = decode_text(model.generate(context, max_new_tokens=100)[0].tolist())
        print(f"Generated text: {generated_chars}")
        if prompt.lower() == 'exit':
            break
