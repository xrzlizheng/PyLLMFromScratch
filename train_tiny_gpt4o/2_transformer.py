#!/usr/bin/env python
# coding: utf-8
"""
@Author: lizheng
@Date: 2025-04-02
@Description: https://lizheng.blog.csdn.net/article/details/147091139?spm=1011.2415.3001.5331
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import math
import os

# For reproducibility (optional, but good practice)
torch.manual_seed(1337)

print(f"PyTorch version: {torch.__version__}")
print("Libraries imported.")



corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""

print(f"Training corpus defined (length: {len(corpus_raw)} characters).")


chars = sorted(list(set(corpus_raw)))
vocab_size = len(chars)

# Create character-to-integer mapping (encoding)
char_to_int = { ch:i for i,ch in enumerate(chars) }

# Create integer-to-character mapping (decoding)
int_to_char = { i:ch for i,ch in enumerate(chars) }

print(f"Created character vocabulary of size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

encoded_corpus = [char_to_int[ch] for ch in corpus_raw]

# Convert the list into a PyTorch tensor
full_data_sequence = torch.tensor(encoded_corpus, dtype=torch.long)

print(f"Encoded corpus into a tensor of shape: {full_data_sequence.shape}")
# print(f"First 100 encoded token IDs: {full_data_sequence[:100].tolist()}") # Optional


# Define Model Hyperparameters (using calculated vocab_size)
# vocab_size = vocab_size # Already defined from data
d_model = 64         # Embedding dimension (increased slightly for characters)
n_heads = 4          # Number of attention heads
n_layers = 3         # Number of Transformer blocks
d_ff = d_model * 4   # Dimension of the feed-forward inner layer
block_size = 32      # Maximum context length (sequence length)
# dropout_rate = 0.1 # Omitting dropout layers for inline simplicity

# Define Training Hyperparameters
learning_rate = 3e-4 # Slightly smaller LR often better for AdamW
batch_size = 16      # Process 16 sequences per step
epochs = 5000        # Increase epochs for character-level model to see learning
eval_interval = 500 # How often to print loss

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure d_model is divisible by n_heads
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
d_k = d_model // n_heads # Dimension of keys/queries/values per head

print(f"Hyperparameters defined:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  n_heads: {n_heads}")
print(f"  d_k (dim per head): {d_k}")
print(f"  n_layers: {n_layers}")
print(f"  d_ff: {d_ff}")
print(f"  block_size: {block_size}")
print(f"  learning_rate: {learning_rate}")
print(f"  batch_size: {batch_size}")
print(f"  epochs: {epochs}")
print(f"  Using device: {device}")

all_x = []
all_y = []

# Iterate through the encoded corpus tensor to extract overlapping sequences
# We need to stop early enough so that we can always get a target sequence of the same length
num_total_tokens = len(full_data_sequence)
for i in range(num_total_tokens - block_size):
    # Extract the input sequence chunk of length block_size
    x_chunk = full_data_sequence[i : i + block_size]
    # Extract the target sequence chunk (shifted one position to the right)
    y_chunk = full_data_sequence[i + 1 : i + block_size + 1]
    
    # Append the chunks to our lists
    all_x.append(x_chunk)
    all_y.append(y_chunk)

train_x = torch.stack(all_x)
train_y = torch.stack(all_y)

num_sequences_available = train_x.shape[0]
print(f"Created {num_sequences_available} overlapping input/target sequence pairs.")
print(f"Shape of train_x: {train_x.shape}")
print(f"Shape of train_y: {train_y.shape}")



# Check if we have enough sequences for the desired batch size
if num_sequences_available < batch_size:
    print(f"Warning: Number of sequences ({num_sequences_available}) is less than batch size ({batch_size}). Adjusting batch size.")
    batch_size = num_sequences_available

print(f"Data ready for training. Will sample batches of size {batch_size} randomly.")



# Initialize the token embedding table (lookup table)
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)

print(f"Initialized Token Embedding Layer (Vocab: {vocab_size}, Dim: {d_model}). Device: {device}")




# Precompute the Sinusoidal Positional Encoding matrix
print("Step 2.2: Creating Positional Encoding matrix...")

# Matrix to store encodings: Shape (block_size, d_model)
positional_encoding = torch.zeros(block_size, d_model, device=device)

# Position indices (0 to block_size-1): Shape (block_size, 1)
position = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)

# Dimension indices (0, 2, 4, ...): Shape (d_model/2)
div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
# Denominator term: 1 / (10000^(2i / d_model))
div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))

# Calculate sine for even dimensions
positional_encoding[:, 0::2] = torch.sin(position * div_term)

# Calculate cosine for odd dimensions
positional_encoding[:, 1::2] = torch.cos(position * div_term)

# Add batch dimension: Shape (1, block_size, d_model)
positional_encoding = positional_encoding.unsqueeze(0)

print(f"  Positional Encoding matrix created with shape: {positional_encoding.shape}. Device: {device}")





print(f"Step 2.3: Initializing components for {n_layers} Transformer layers...")

# Lists to store layers for each Transformer block
layer_norms_1 = []      # LayerNorm after MHA
layer_norms_2 = []      # LayerNorm after FFN
mha_qkv_linears = []    # Combined Linear layer for Q, K, V projections
mha_output_linears = [] # Output Linear layer for MHA
ffn_linear_1 = []       # First linear layer in FFN
ffn_linear_2 = []       # Second linear layer in FFN

# Loop through the number of layers
for i in range(n_layers):
    # Layer Normalization 1 (for post-MHA residual)
    ln1 = nn.LayerNorm(d_model).to(device)
    layer_norms_1.append(ln1)

    # Multi-Head Attention: Combined QKV projection layer
    qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False).to(device) # Often bias=False here
    mha_qkv_linears.append(qkv_linear)

    # Multi-Head Attention: Output projection layer
    output_linear = nn.Linear(d_model, d_model).to(device)
    mha_output_linears.append(output_linear)

    # Layer Normalization 2 (for post-FFN residual)
    ln2 = nn.LayerNorm(d_model).to(device)
    layer_norms_2.append(ln2)
    
    # Position-wise Feed-Forward Network: First linear layer
    lin1 = nn.Linear(d_model, d_ff).to(device)
    ffn_linear_1.append(lin1)
    
    # Position-wise Feed-Forward Network: Second linear layer
    lin2 = nn.Linear(d_ff, d_model).to(device)
    ffn_linear_2.append(lin2)
    
    print(f"  Initialized components for Layer {i+1}/{n_layers}.")

print(f"Finished initializing components for {n_layers} layers.")



# Final Layer Normalization
final_layer_norm = nn.LayerNorm(d_model).to(device)
print(f"  Initialized Final LayerNorm. Device: {device}")

# Final Linear Layer (language modeling head)
output_linear_layer = nn.Linear(d_model, vocab_size).to(device)
print(f"  Initialized Output Linear Layer (to vocab size {vocab_size}). Device: {device}")

# Define the loss function
criterion = nn.CrossEntropyLoss()

print(f"Step 4.1: Loss function defined: {type(criterion).__name__}")





# Gather all model parameters requiring gradients
all_model_parameters = list(token_embedding_table.parameters())
for i in range(n_layers):
    all_model_parameters.extend(list(layer_norms_1[i].parameters()))
    all_model_parameters.extend(list(mha_qkv_linears[i].parameters()))
    all_model_parameters.extend(list(mha_output_linears[i].parameters()))
    all_model_parameters.extend(list(layer_norms_2[i].parameters()))
    all_model_parameters.extend(list(ffn_linear_1[i].parameters()))
    all_model_parameters.extend(list(ffn_linear_2[i].parameters()))
all_model_parameters.extend(list(final_layer_norm.parameters()))
all_model_parameters.extend(list(output_linear_layer.parameters()))

# Define the AdamW optimizer
optimizer = optim.AdamW(all_model_parameters, lr=learning_rate)

print(f"Step 4.2: Optimizer defined: {type(optimizer).__name__}")
print(f"  Managing {len(all_model_parameters)} parameter groups/tensors.")

# Create the lower triangular mask for self-attention ONCE, outside the loop
# Shape: (1, 1, block_size, block_size)
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)





print(f"\nStep 4.3: Starting Training Loop for {epochs} epochs...")

# List to track losses
losses = []

# Set layers to training mode (e.g., for potential dropout, though omitted here)
# This doesn't really do anything without dropout/batchnorm, but good practice
for i in range(n_layers):
    layer_norms_1[i].train()
    mha_qkv_linears[i].train()
    mha_output_linears[i].train()
    layer_norms_2[i].train()
    ffn_linear_1[i].train()
    ffn_linear_2[i].train()
final_layer_norm.train()
output_linear_layer.train()
token_embedding_table.train()

# Training loop
for epoch in range(epochs):
    
    # --- 1. Batch Selection --- 
    indices = torch.randint(0, num_sequences_available, (batch_size,))
    xb = train_x[indices].to(device) # Input batch shape: (B, T)
    yb = train_y[indices].to(device) # Target batch shape: (B, T)
    
    # --- 2. Forward Pass (Inline execution) --- 
    B, T = xb.shape # B = batch_size, T = block_size
    C = d_model     # Embedding dimension
    
    # Step 3.1: Embedding + Positional Encoding
    token_embed = token_embedding_table(xb) # (B, T, C)
    pos_enc_slice = positional_encoding[:, :T, :] # (1, T, C)
    x = token_embed + pos_enc_slice # (B, T, C)
    
    # Step 3.2: Transformer Blocks
    for i in range(n_layers):
        # Input to this block
        x_input_block = x 
        
        # --- MHA --- 
        # Apply LayerNorm *before* MHA (Pre-LN variant - common)
        x_ln1 = layer_norms_1[i](x_input_block)
        # QKV projection
        qkv = mha_qkv_linears[i](x_ln1) # (B, T, 3*C)
        # Split heads
        qkv = qkv.view(B, T, n_heads, 3 * d_k).permute(0, 2, 1, 3) # (B, n_heads, T, 3*d_k)
        q, k, v = qkv.chunk(3, dim=-1) # (B, n_heads, T, d_k)
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * (d_k ** -0.5) # (B, n_heads, T, T)
        # Apply Causal Mask (use the pre-computed mask sliced to T)
        attn_scores_masked = attn_scores.masked_fill(causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attn_scores_masked, dim=-1) # (B, n_heads, T, T)
        # Attention output
        attn_output = attention_weights @ v # (B, n_heads, T, d_k)
        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C) # (B, T, C)
        # Output projection
        mha_result = mha_output_linears[i](attn_output) # (B, T, C)
        # Add & Norm 1 (Residual connection adds output to original input)
        x = x_input_block + mha_result # Residual connection 1
        # Note: We moved LN1 to *before* MHA (Pre-LN)
        
        # --- FFN --- 
        # Input to FFN
        x_input_ffn = x 
        # Apply LayerNorm *before* FFN (Pre-LN variant)
        x_ln2 = layer_norms_2[i](x_input_ffn)
        # FFN layers
        ffn_hidden = ffn_linear_1[i](x_ln2) # (B, T, d_ff)
        ffn_activated = F.relu(ffn_hidden)
        ffn_output = ffn_linear_2[i](ffn_activated) # (B, T, C)
        # Add & Norm 2 (Residual connection adds output to FFN input)
        x = x_input_ffn + ffn_output # Residual connection 2
        # Note: We moved LN2 to *before* FFN (Pre-LN)
        # Output 'x' of this block becomes input 'x_input_block' for the next block
        
    # Step 3.3: Final Layers (After loop)
    # Apply final LayerNorm (Pre-LN style, applied before final projection)
    final_norm_output = final_layer_norm(x) # (B, T, C)
    logits = output_linear_layer(final_norm_output) # (B, T, vocab_size)
    
    # --- 3. Calculate Loss --- 
    B_loss, T_loss, V_loss = logits.shape
    logits_for_loss = logits.view(B_loss * T_loss, V_loss) 
    targets_for_loss = yb.view(B_loss * T_loss)
    loss = criterion(logits_for_loss, targets_for_loss)
    
    # --- 4. Zero Gradients --- 
    optimizer.zero_grad()
    
    # --- 5. Backward Pass --- 
    loss.backward()
    
    # --- 6. Update Parameters --- 
    optimizer.step()
    
    # --- Logging --- 
    current_loss = loss.item()
    losses.append(current_loss)
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")

print("--- Training Loop Completed ---")




print("\n--- Step 5: Text Generation ---")

# Seed character(s)
seed_chars = "t"
# Convert seed characters to token IDs
seed_ids = [char_to_int[ch] for ch in seed_chars]

# Create the initial context tensor
# Shape: (1, len(seed_ids)) -> Batch dimension = 1
generated_sequence = torch.tensor([seed_ids], dtype=torch.long, device=device)
print(f"Initial seed sequence: '{seed_chars}' -> {generated_sequence.tolist()}")

# Define how many new tokens (characters) to generate
num_tokens_to_generate = 200 
print(f"Generating {num_tokens_to_generate} new tokens...")




for i in range(n_layers):
    layer_norms_1[i].eval()
    mha_qkv_linears[i].eval()
    mha_output_linears[i].eval()
    layer_norms_2[i].eval()
    ffn_linear_1[i].eval()
    ffn_linear_2[i].eval()
final_layer_norm.eval()
output_linear_layer.eval()
token_embedding_table.eval()

# Disable gradient calculations for generation
with torch.no_grad():
    # Loop to generate tokens one by one
    for _ in range(num_tokens_to_generate):
        # --- 1. Prepare Input Context --- 
        # Take the last block_size tokens as context
        current_context = generated_sequence[:, -block_size:] # Shape: (1, min(current_len, block_size))
        B_gen, T_gen = current_context.shape 
        C_gen = d_model
        
        # --- 2. Forward Pass --- 
        # Embedding + Positional Encoding
        token_embed_gen = token_embedding_table(current_context) # (B_gen, T_gen, C_gen)
        pos_enc_slice_gen = positional_encoding[:, :T_gen, :] 
        x_gen = token_embed_gen + pos_enc_slice_gen # (B_gen, T_gen, C_gen)
        
        # Transformer Blocks
        for i in range(n_layers):
            x_input_block_gen = x_gen
            # Pre-LN MHA
            x_ln1_gen = layer_norms_1[i](x_input_block_gen)
            qkv_gen = mha_qkv_linears[i](x_ln1_gen)
            qkv_gen = qkv_gen.view(B_gen, T_gen, n_heads, 3 * d_k).permute(0, 2, 1, 3)
            q_gen, k_gen, v_gen = qkv_gen.chunk(3, dim=-1)
            attn_scores_gen = (q_gen @ k_gen.transpose(-2, -1)) * (d_k ** -0.5)
            # Use the pre-computed mask sliced to the current context length T_gen
            attn_scores_masked_gen = attn_scores_gen.masked_fill(causal_mask[:,:,:T_gen,:T_gen] == 0, float('-inf'))
            attention_weights_gen = F.softmax(attn_scores_masked_gen, dim=-1)
            attn_output_gen = attention_weights_gen @ v_gen
            attn_output_gen = attn_output_gen.permute(0, 2, 1, 3).contiguous().view(B_gen, T_gen, C_gen)
            mha_result_gen = mha_output_linears[i](attn_output_gen)
            x_gen = x_input_block_gen + mha_result_gen # Residual 1
            # Pre-LN FFN
            x_input_ffn_gen = x_gen
            x_ln2_gen = layer_norms_2[i](x_input_ffn_gen)
            ffn_hidden_gen = ffn_linear_1[i](x_ln2_gen)
            ffn_activated_gen = F.relu(ffn_hidden_gen)
            ffn_output_gen = ffn_linear_2[i](ffn_activated_gen)
            x_gen = x_input_ffn_gen + ffn_output_gen # Residual 2
            
        # Final Layers
        final_norm_output_gen = final_layer_norm(x_gen)
        logits_gen = output_linear_layer(final_norm_output_gen) # (B_gen, T_gen, vocab_size)
        
        # --- 3. Get Logits for Last Time Step --- 
        logits_last_token = logits_gen[:, -1, :] # Shape: (B_gen, vocab_size)
        
        # --- 4. Apply Softmax --- 
        probs = F.softmax(logits_last_token, dim=-1) # Shape: (B_gen, vocab_size)
        
        # --- 5. Sample Next Token --- 
        next_token = torch.multinomial(probs, num_samples=1) # Shape: (B_gen, 1)
        
        # --- 6. Append Sampled Token --- 
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

print("\n--- Generation Complete ---")



final_generated_ids = generated_sequence[0].tolist()

# Decode the list of IDs back into a string
decoded_text = ''.join([int_to_char[id] for id in final_generated_ids])

print(f"\nFinal Generated Text (including seed):")
print(decoded_text)


os.makedirs('saved_models', exist_ok=True)

# Create a state dictionary to hold all model parameters
state_dict = {
    'token_embedding_table': token_embedding_table.state_dict(),
    'positional_encoding': positional_encoding,  # This is not a parameter, just a tensor
    'layer_norms_1': [ln.state_dict() for ln in layer_norms_1],
    'mha_qkv_linears': [linear.state_dict() for linear in mha_qkv_linears],
    'mha_output_linears': [linear.state_dict() for linear in mha_output_linears],
    'layer_norms_2': [ln.state_dict() for ln in layer_norms_2],
    'ffn_linear_1': [linear.state_dict() for linear in ffn_linear_1],
    'ffn_linear_2': [linear.state_dict() for linear in ffn_linear_2],
    'final_layer_norm': final_layer_norm.state_dict(),
    'output_linear_layer': output_linear_layer.state_dict(),
    # Save hyperparameters for model reconstruction
    'config': {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'block_size': block_size
    },
    # Save tokenizer info for text generation
    'tokenizer': {
        'char_to_int': char_to_int,
        'int_to_char': int_to_char
    }
}

# Save the state dictionary
torch.save(state_dict, 'saved_models/transformer_model.pt')
print("Model saved successfully to 'saved_models/transformer_model.pt'")


# To load the model later, you would do:




# Load the saved state dictionary
loaded_state_dict = torch.load('saved_models/transformer_model.pt', map_location=device)

# Extract configuration and tokenizer info
config = loaded_state_dict['config']
vocab_size = config['vocab_size']
d_model = config['d_model']
n_heads = config['n_heads']
n_layers = config['n_layers']
d_ff = config['d_ff']
block_size = config['block_size']
d_k = d_model // n_heads

char_to_int = loaded_state_dict['tokenizer']['char_to_int']
int_to_char = loaded_state_dict['tokenizer']['int_to_char']

# Recreate the model components
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)
token_embedding_table.load_state_dict(loaded_state_dict['token_embedding_table'])

positional_encoding = loaded_state_dict['positional_encoding'].to(device)

# Initialize the layer lists
layer_norms_1 = []
mha_qkv_linears = []
mha_output_linears = []
layer_norms_2 = []
ffn_linear_1 = []
ffn_linear_2 = []

# Load each layer's components
for i in range(n_layers):
    # Layer norm 1
    ln1 = nn.LayerNorm(d_model).to(device)
    ln1.load_state_dict(loaded_state_dict['layer_norms_1'][i])
    layer_norms_1.append(ln1)
    
    # MHA QKV linear
    qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False).to(device)
    qkv_linear.load_state_dict(loaded_state_dict['mha_qkv_linears'][i])
    mha_qkv_linears.append(qkv_linear)
    
    # MHA output linear
    output_linear = nn.Linear(d_model, d_model).to(device)
    output_linear.load_state_dict(loaded_state_dict['mha_output_linears'][i])
    mha_output_linears.append(output_linear)
    
    # Layer norm 2
    ln2 = nn.LayerNorm(d_model).to(device)
    ln2.load_state_dict(loaded_state_dict['layer_norms_2'][i])
    layer_norms_2.append(ln2)
    
    # FFN linear 1
    lin1 = nn.Linear(d_model, d_ff).to(device)
    lin1.load_state_dict(loaded_state_dict['ffn_linear_1'][i])
    ffn_linear_1.append(lin1)
    
    # FFN linear 2
    lin2 = nn.Linear(d_ff, d_model).to(device)
    lin2.load_state_dict(loaded_state_dict['ffn_linear_2'][i])
    ffn_linear_2.append(lin2)

# Final layer norm
final_layer_norm = nn.LayerNorm(d_model).to(device)
final_layer_norm.load_state_dict(loaded_state_dict['final_layer_norm'])

# Output linear layer
output_linear_layer = nn.Linear(d_model, vocab_size).to(device)
output_linear_layer.load_state_dict(loaded_state_dict['output_linear_layer'])

print("Model loaded successfully!")


