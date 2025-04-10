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
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import math
import os
import numpy as np # For creating dummy images

# For reproducibility (optional, but good practice)
torch.manual_seed(42) # Use a different seed for variation
np.random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print("Libraries imported.")

# --- Device Configuration ---
# Theory: Set the device (GPU if available, else CPU) for tensor operations.
# This ensures that models and data are processed on the same hardware.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")



model_load_path = 'saved_models/transformer_model.pt'
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_load_path}. Please ensure 'transformer2.ipynb' was run and saved the model.")

loaded_state_dict = torch.load(model_load_path, map_location=device)
print(f"Loaded state dictionary from '{model_load_path}'.")


config = loaded_state_dict['config']
loaded_vocab_size = config['vocab_size']
d_model = config['d_model']
n_heads = config['n_heads']
n_layers = config['n_layers']
d_ff = config['d_ff']
loaded_block_size = config['block_size'] # Max sequence length for text model
d_k = d_model // n_heads

char_to_int = loaded_state_dict['tokenizer']['char_to_int']
int_to_char = loaded_state_dict['tokenizer']['int_to_char']

print("Extracted model configuration and tokenizer:")
print(f"  Loaded vocab_size: {loaded_vocab_size}")
print(f"  d_model: {d_model}")
print(f"  n_layers: {n_layers}")
print(f"  n_heads: {n_heads}")
print(f"  d_ff: {d_ff}")
print(f"  Loaded block_size: {loaded_block_size}")






# --- Define Special Tokens ---
img_token = "<IMG>"
pad_token = "<PAD>"
eos_token = "<EOS>" # End-of-Sentence/Sequence
special_tokens = [img_token, pad_token, eos_token]

current_vocab_size = loaded_vocab_size
for token in special_tokens:
    if token not in char_to_int:
        char_to_int[token] = current_vocab_size
        int_to_char[current_vocab_size] = token
        current_vocab_size += 1

# Update vocab_size
vocab_size = current_vocab_size
pad_token_id = char_to_int[pad_token] # Store the ID for later use

print(f"Added special tokens: {special_tokens}")
print(f"Updated vocabulary size: {vocab_size}")
print(f"PAD token ID: {pad_token_id}")


sample_data_dir = "sample_multimodal_data"
os.makedirs(sample_data_dir, exist_ok=True)

image_paths = {
    "red": os.path.join(sample_data_dir, "red_square.png"),
    "blue": os.path.join(sample_data_dir, "blue_square.png"),
    "green": os.path.join(sample_data_dir, "green_circle.png") # Let's add a shape difference
}

# Create Red Square
img_red = Image.new('RGB', (64, 64), color = 'red')
img_red.save(image_paths["red"])
# Create Blue Square
img_blue = Image.new('RGB', (64, 64), color = 'blue')
img_blue.save(image_paths["blue"])
# Create Green Circle (approximate with PIL draw)
img_green = Image.new('RGB', (64, 64), color = 'white')
from PIL import ImageDraw
draw = ImageDraw.Draw(img_green)
draw.ellipse((4, 4, 60, 60), fill='green', outline='green')
img_green.save(image_paths["green"])

print(f"Created dummy images in '{sample_data_dir}'.")


sample_training_data = [
    {"image_path": image_paths["red"], "prompt": "What color is the shape?", "response": "red." + eos_token},
    {"image_path": image_paths["blue"], "prompt": "Describe the image.", "response": "a blue square." + eos_token},
    {"image_path": image_paths["green"], "prompt": "What shape is shown?", "response": "a green circle." + eos_token},
    {"image_path": image_paths["red"], "prompt": "Is it a circle?", "response": "no, it is a square." + eos_token},
    {"image_path": image_paths["blue"], "prompt": "What is the main color?", "response": "blue." + eos_token},
    {"image_path": image_paths["green"], "prompt": "Describe this.", "response": "a circle, it is green." + eos_token}
]

num_samples = len(sample_training_data)
print(f"Defined {num_samples} sample multi-modal data points.")
# print(f"Sample 0: {sample_training_data[0]}")








print(f"Loaded ResNet-18 feature extractor.")
print(f"  Output feature dimension: {vision_feature_dim}") # Should be 512 for ResNet-18
print(f"  Vision model set to evaluation mode on device: {device}")



image_transforms = transforms.Compose([
    transforms.Resize(256),            # Resize smaller edge to 256
    transforms.CenterCrop(224),        # Crop center 224x224 square
    transforms.ToTensor(),             # Convert PIL Image to FloatTensor (0-1 range)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std dev
])

print("Defined image preprocessing pipeline (Resize, Crop, ToTensor, Normalize).")






block_size = 64 # Max length for combined sequence (e.g., 1 IMG + prompt + response)
print(f"  Set combined block_size: {block_size}")

# --- Number of Image Tokens ---
# Theory: How many token positions will represent the image? We'll use 1 for simplicity.
num_img_tokens = 1
print(f"  Using {num_img_tokens} <IMG> token(s) to represent image features.")

# --- Training Parameters ---
# Re-state or adjust training parameters if needed
learning_rate = 3e-4 # Keep the same AdamW learning rate
batch_size = 4 # Reduce batch size due to potentially larger memory footprint
epochs = 2000  # Increase epochs further for multi-modal learning
eval_interval = 500

print(f"  Updated Training Params: LR={learning_rate}, BatchSize={batch_size}, Epochs={epochs}")

# Ensure block_size is sufficient
min_req_block_size = num_img_tokens + max(len(d["prompt"]) + len(d["response"]) for d in sample_training_data) + 1 # +1 for safety/EOS
print(f"  Max sequence length in sample data (approx): {min_req_block_size}")
if block_size < min_req_block_size:
     print(f"Warning: block_size ({block_size}) might be too small for the longest sample sequence ({min_req_block_size}). Consider increasing it.")


causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)
print(f"  Recreated causal mask for new block_size={block_size}")






extracted_image_features = {} # Dict to store {image_path: feature_tensor}

# --- Loop Through Unique Image Paths ---
unique_image_paths = set(d["image_path"] for d in sample_training_data)
print(f"Found {len(unique_image_paths)} unique images to process.")

for img_path in unique_image_paths:
    # --- Load Image ---
    # Theory: Open the image file using Pillow.
    try:
        img = Image.open(img_path).convert('RGB') # Ensure image is RGB
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}. Skipping.")
        continue

    # --- Apply Transformations ---
    # Theory: Apply the predefined preprocessing pipeline (resize, crop, tensor, normalize).
    # Unsqueeze(0) adds a batch dimension (B=1) as expected by the vision model.
    img_tensor = image_transforms(img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)

    # --- Extract Features ---
    # Theory: Pass the preprocessed image tensor through the vision model.
    # Use torch.no_grad() as we are not training the vision model here.
    with torch.no_grad():
        feature_vector = vision_model(img_tensor) # Shape: (1, vision_feature_dim)

    # --- Store Features ---
    extracted_image_features[img_path] = feature_vector.squeeze(0) # Remove batch dim, store (feature_dim,)
    print(f"  Extracted features for '{os.path.basename(img_path)}', shape: {extracted_image_features[img_path].shape}")

print("Finished extracting image features for all unique sample images.")






# First, extend vocabulary with any new characters
current_vocab_size = vocab_size  # Start from current vocabulary size
all_chars = set()

# Collect all unique characters from prompts and responses
for sample in sample_training_data:
    all_chars.update(sample["prompt"])
    # Remove EOS token from response before collecting chars
    response_text = sample["response"]
    if response_text.endswith(eos_token):
        response_text = response_text[:-len(eos_token)]
    all_chars.update(response_text)

# Add any new characters to vocabulary
new_chars_added = 0
for char in all_chars:
    if char not in char_to_int:
        char_to_int[char] = current_vocab_size
        int_to_char[current_vocab_size] = char
        current_vocab_size += 1
        new_chars_added += 1

vocab_size = current_vocab_size  # Update vocab size
print(f"Added {new_chars_added} new characters to vocabulary. New vocab_size: {vocab_size}")

# Now tokenize with the extended vocabulary
tokenized_samples = []
for sample in sample_training_data:
    # --- Tokenize Prompt ---
    prompt_ids = [char_to_int[ch] for ch in sample["prompt"]]

    # --- Tokenize Response ---
    response_text = sample["response"]
    if response_text.endswith(eos_token):
        response_text_without_eos = response_text[:-len(eos_token)]
        response_ids = [char_to_int[ch] for ch in response_text_without_eos] + [char_to_int[eos_token]]
    else:
        response_ids = [char_to_int[ch] for ch in response_text]

    tokenized_samples.append({
        "image_path": sample["image_path"],
        "prompt_ids": prompt_ids,
        "response_ids": response_ids
    })

print(f"Tokenized text for all {len(tokenized_samples)} samples.")



prepared_sequences = []
ignore_index = -100 # Common ignore index for CrossEntropyLoss

for sample in tokenized_samples:
    # --- Construct Input Sequence IDs ---
    img_ids = [char_to_int[img_token]] * num_img_tokens
    input_ids_no_pad = img_ids + sample["prompt_ids"] + sample["response_ids"][:-1] # Input predicts response


    target_ids_no_pad = ([ignore_index] * len(img_ids)) + 
                         ([ignore_index] * len(sample["prompt_ids"])) +
                         sample["response_ids"]

    # --- Padding ---
    current_len = len(input_ids_no_pad)
    pad_len = block_size - current_len

    if pad_len < 0:
        print(f"Warning: Sample sequence length ({current_len}) exceeds block_size ({block_size}). Truncating.")
        input_ids = input_ids_no_pad[:block_size]
        target_ids = target_ids_no_pad[:block_size]
        pad_len = 0 # No padding needed after truncation
        current_len = block_size
    else:
        input_ids = input_ids_no_pad + ([pad_token_id] * pad_len)
        target_ids = target_ids_no_pad + ([ignore_index] * pad_len) # Pad targets with ignore_index


    attention_mask = ([1] * current_len) + ([0] * pad_len)

    # --- Store ---
    prepared_sequences.append({
        "image_path": sample["image_path"],
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long) # Or float for some implementations
    })


all_input_ids = torch.stack([s['input_ids'] for s in prepared_sequences])
all_target_ids = torch.stack([s['target_ids'] for s in prepared_sequences])
all_attention_masks = torch.stack([s['attention_mask'] for s in prepared_sequences])
# Keep image paths associated for retrieving features during batching
all_image_paths = [s['image_path'] for s in prepared_sequences]

num_sequences_available = all_input_ids.shape[0]
print(f"Created {num_sequences_available} padded sequences with targets and masks.")
print(f"  Input IDs shape: {all_input_ids.shape}") # (num_samples, block_size)
print(f"  Target IDs shape: {all_target_ids.shape}") # (num_samples, block_size)
print(f"  Attention Mask shape: {all_attention_masks.shape}") # (num_samples, block_size)



max_seq_len = 64  # This should match what your model expects

# Ensure all sequences are properly sized
for i in range(len(tokenized_samples)):
    # Ensure response_ids don't exceed max length
    if len(tokenized_samples[i]["response_ids"]) > max_seq_len:
        tokenized_samples[i]["response_ids"] = tokenized_samples[i]["response_ids"][:max_seq_len]
    
    # Also check prompt_ids if needed
    if len(tokenized_samples[i]["prompt_ids"]) > max_seq_len:
        tokenized_samples[i]["prompt_ids"] = tokenized_samples[i]["prompt_ids"][:max_seq_len]

print(f"All sequences adjusted to maximum length of {max_seq_len}")







# Check if batch size is feasible
if num_sequences_available < batch_size:
    print(f"Warning: Number of sequences ({num_sequences_available}) is less than batch size ({batch_size}). Adjusting batch size.")
    batch_size = num_sequences_available

print(f"Data ready for training. Will sample batches of size {batch_size} randomly.")





# --- Token Embedding Table ---
new_token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)
# Load weights for the original vocabulary part
original_weights = loaded_state_dict['token_embedding_table']['weight'][:loaded_vocab_size, :]
with torch.no_grad():
    new_token_embedding_table.weight[:loaded_vocab_size, :] = original_weights
    # New tokens (<IMG>, <PAD>, <EOS>) are randomly initialized by default, which is fine.
token_embedding_table = new_token_embedding_table # Replace the old variable
print(f"  Re-initialized Token Embedding Table, shape: {token_embedding_table.weight.shape}")

# --- Output Linear Layer ---
new_output_linear_layer = nn.Linear(d_model, vocab_size).to(device)
# Load weights and biases for the original vocabulary part
original_out_weight = loaded_state_dict['output_linear_layer']['weight'][:loaded_vocab_size, :]
original_out_bias = loaded_state_dict['output_linear_layer']['bias'][:loaded_vocab_size]
with torch.no_grad():
    new_output_linear_layer.weight[:loaded_vocab_size, :] = original_out_weight
    new_output_linear_layer.bias[:loaded_vocab_size] = original_out_bias
    # Weights/biases for new tokens are randomly initialized.
output_linear_layer = new_output_linear_layer # Replace the old variable
print(f"  Re-initialized Output Linear Layer, weight shape: {output_linear_layer.weight.shape}")






vision_projection_layer = nn.Linear(vision_feature_dim, d_model).to(device)

print(f"  Initialized Vision Projection Layer: {vision_feature_dim} -> {d_model}. Device: {device}")



# Lists to store layers for each Transformer block
layer_norms_1 = []
layer_norms_2 = []
mha_qkv_linears = []
mha_output_linears = []
ffn_linear_1 = []
ffn_linear_2 = []

# Load components for each layer from the state dict
for i in range(n_layers):
    # LayerNorm 1
    ln1 = nn.LayerNorm(d_model).to(device)
    ln1.load_state_dict(loaded_state_dict['layer_norms_1'][i])
    layer_norms_1.append(ln1)

    # MHA QKV Linear
    qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False).to(device)
    qkv_linear.load_state_dict(loaded_state_dict['mha_qkv_linears'][i])
    mha_qkv_linears.append(qkv_linear)

    # MHA Output Linear
    output_linear_mha = nn.Linear(d_model, d_model).to(device)
    output_linear_mha.load_state_dict(loaded_state_dict['mha_output_linears'][i])
    mha_output_linears.append(output_linear_mha) # Renamed to avoid conflict

    # LayerNorm 2
    ln2 = nn.LayerNorm(d_model).to(device)
    ln2.load_state_dict(loaded_state_dict['layer_norms_2'][i])
    layer_norms_2.append(ln2)

    # FFN Linear 1
    lin1 = nn.Linear(d_model, d_ff).to(device)
    lin1.load_state_dict(loaded_state_dict['ffn_linear_1'][i])
    ffn_linear_1.append(lin1)

    # FFN Linear 2
    lin2 = nn.Linear(d_ff, d_model).to(device)
    lin2.load_state_dict(loaded_state_dict['ffn_linear_2'][i])
    ffn_linear_2.append(lin2)

    print(f"  Loaded components for Layer {i+1}/{n_layers}.")

# Load Final LayerNorm
final_layer_norm = nn.LayerNorm(d_model).to(device)
final_layer_norm.load_state_dict(loaded_state_dict['final_layer_norm'])
print("  Loaded Final LayerNorm.")

# Load Positional Encoding (not a parameter, but needed)
positional_encoding = loaded_state_dict['positional_encoding'].to(device)
# Adjust positional encoding if block_size changed significantly
if positional_encoding.shape[1] != block_size:
     print(f"Warning: Loaded positional encoding size ({positional_encoding.shape[1]}) != new block_size ({block_size}). Recomputing.")
     # Recompute PE for the new block_size (copy code from Step 0.5 of transformer2.ipynb if needed)
     # Or simply slice/pad the existing one if change is small (less accurate)
     # For simplicity, we'll just recreate it using the code from transformer2
     new_pe = torch.zeros(block_size, d_model, device=device)
     position = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)
     div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
     div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))
     new_pe[:, 0::2] = torch.sin(position * div_term)
     new_pe[:, 1::2] = torch.cos(position * div_term)
     positional_encoding = new_pe.unsqueeze(0) # Add batch dim
     print(f"  Recomputed Positional Encoding matrix, shape: {positional_encoding.shape}")


print("Finished loading existing model components.")







all_trainable_parameters = list(token_embedding_table.parameters())
all_trainable_parameters.extend(list(vision_projection_layer.parameters())) # Add new layer
for i in range(n_layers):
    all_trainable_parameters.extend(list(layer_norms_1[i].parameters()))
    all_trainable_parameters.extend(list(mha_qkv_linears[i].parameters()))
    all_trainable_parameters.extend(list(mha_output_linears[i].parameters())) # Use correct name
    all_trainable_parameters.extend(list(layer_norms_2[i].parameters()))
    all_trainable_parameters.extend(list(ffn_linear_1[i].parameters()))
    all_trainable_parameters.extend(list(ffn_linear_2[i].parameters()))
all_trainable_parameters.extend(list(final_layer_norm.parameters()))
all_trainable_parameters.extend(list(output_linear_layer.parameters()))

# --- Define Optimizer ---
optimizer = optim.AdamW(all_trainable_parameters, lr=learning_rate)
print(f"  Optimizer defined: AdamW with lr={learning_rate}")
print(f"  Managing {len(all_trainable_parameters)} parameter groups/tensors.")

# --- Define Loss Function ---
# Theory: Use CrossEntropyLoss with ignore_index set to ignore padding AND prompt tokens.
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
print(f"  Loss function defined: CrossEntropyLoss (ignore_index={ignore_index})")


# List to track losses
losses = []

# --- Set Layers to Training Mode ---
# Theory: Ensure layers like LayerNorm are in training mode. Vision model remains in eval.
token_embedding_table.train()
vision_projection_layer.train()
for i in range(n_layers):
    layer_norms_1[i].train()
    mha_qkv_linears[i].train()
    mha_output_linears[i].train() # Use correct name
    layer_norms_2[i].train()
    ffn_linear_1[i].train()
    ffn_linear_2[i].train()
final_layer_norm.train()
output_linear_layer.train()
# vision_model remains in eval() mode

# --- Training Loop ---
for epoch in range(epochs):

    # --- 1. Batch Selection ---
    indices = torch.randint(0, num_sequences_available, (batch_size,))
    # Retrieve data for the batch
    xb_ids = all_input_ids[indices].to(device)          # (B, T)
    yb_ids = all_target_ids[indices].to(device)          # (B, T)
    batch_masks = all_attention_masks[indices].to(device) # (B, T) - Basic padding mask
    batch_img_paths = [all_image_paths[i] for i in indices.tolist()]
    # Retrieve pre-extracted features and stack them into a batch
    try:
        batch_img_features = torch.stack([extracted_image_features[p] for p in batch_img_paths]).to(device) # (B, vision_feature_dim)
    except KeyError as e:
        print(f"Error: Missing extracted feature for image path {e}. Ensure Step 1.1 completed correctly. Skipping epoch.")
        continue


    # --- 2. Forward Pass (Inline) ---
    B, T = xb_ids.shape # T is block_size
    C = d_model

    # --- Project Image Features ---
    # Theory: Map image features to the Transformer's dimension (d_model).
    projected_img_features = vision_projection_layer(batch_img_features) # (B, C)
    # Unsqueeze to add the sequence dimension for concatenation/addition: (B, 1, C)
    projected_img_features = projected_img_features.unsqueeze(1)
    # Replicate if num_img_tokens > 1 (not needed here as num_img_tokens is 1)
    # if num_img_tokens > 1:
    #     projected_img_features = projected_img_features.repeat(1, num_img_tokens, 1) # (B, num_img_tokens, C)


    # --- Get Text Embeddings ---
    # Theory: Get embeddings for the entire input ID sequence (including <IMG> placeholders).
    text_token_embeddings = token_embedding_table(xb_ids) # (B, T, C)

    # --- Combine Modalities ---
    # Theory: Replace the embedding of the <IMG> token(s) with the projected image features.
    # Since num_img_tokens is 1, we replace the embedding at index 0.
    combined_embeddings = text_token_embeddings.clone() # Avoid modifying original tensor inplace
    combined_embeddings[:, 0:num_img_tokens, :] = projected_img_features # Simple replacement/injection

    # --- Add Positional Encoding ---
    # Theory: Add positional information to the combined sequence. Slice PE to match T.
    pos_enc_slice = positional_encoding[:, :T, :] # (1, T, C)
    x = combined_embeddings + pos_enc_slice # (B, T, C)


    padding_mask_expanded = batch_masks.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
    # Combined mask: causal AND padding. Result is 0 where attention is NOT allowed.
    # Causal mask is (1, 1, T, T). Padding mask is (B, 1, 1, T). Broadcasting applies.
    combined_attn_mask = causal_mask[:,:,:T,:T] * padding_mask_expanded # (B, 1, T, T)


    for i in range(n_layers):
        x_input_block = x
        # Pre-LN MHA
        x_ln1 = layer_norms_1[i](x_input_block)
        qkv = mha_qkv_linears[i](x_ln1)
        qkv = qkv.view(B, T, n_heads, 3 * d_k).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = (q @ k.transpose(-2, -1)) * (d_k ** -0.5) # (B, n_heads, T, T)


        attn_scores_masked = attn_scores.masked_fill(combined_attn_mask == 0, float('-inf'))

        attention_weights = F.softmax(attn_scores_masked, dim=-1) # (B, n_heads, T, T)
        # Handle potential NaNs if a row in softmax is all -inf (e.g., all padding)
        attention_weights = torch.nan_to_num(attention_weights)

        attn_output = attention_weights @ v # (B, n_heads, T, d_k)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        mha_result = mha_output_linears[i](attn_output) # Use correct name
        x = x_input_block + mha_result # Residual 1

        # Pre-LN FFN
        x_input_ffn = x
        x_ln2 = layer_norms_2[i](x_input_ffn)
        ffn_hidden = ffn_linear_1[i](x_ln2)
        ffn_activated = F.relu(ffn_hidden)
        ffn_output = ffn_linear_2[i](ffn_activated)
        x = x_input_ffn + ffn_output # Residual 2

    # --- Final Layers ---
    final_norm_output = final_layer_norm(x) # (B, T, C)
    logits = output_linear_layer(final_norm_output) # (B, T, vocab_size)


    B_loss, T_loss, V_loss = logits.shape
    # print(f"Shapes: logits={logits.shape}, targets={yb_ids.shape}")

    # Ensure targets match the sequence length dimension of logits
    if yb_ids.size(1) != T_loss:

        if yb_ids.size(1) > T_loss:
            # Truncate if targets are longer
            targets_reshaped = yb_ids[:, :T_loss].contiguous().view(-1)
        else:
            # Pad with ignore_index if targets are shorter (shouldn't happen with the fix above)
            padded_targets = torch.full((B_loss, T_loss), ignore_index, device=device)
            padded_targets[:, :yb_ids.size(1)] = yb_ids
            targets_reshaped = padded_targets.view(-1)
    else:
        # Shapes match correctly
        targets_reshaped = yb_ids.view(-1)

    logits_reshaped = logits.view(-1, V_loss)
    loss = criterion(logits_reshaped, targets_reshaped)


    optimizer.zero_grad()


    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()

        optimizer.step()
    else:
        print(f"Warning: Invalid loss detected (NaN or Inf) at epoch {epoch+1}. Skipping optimizer step.")
        loss = None # Set loss to None if invalid

    # --- Logging ---
    if loss is not None:
        current_loss = loss.item()
        losses.append(current_loss)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")
    elif epoch % eval_interval == 0 or epoch == epochs - 1:
         print(f"  Epoch {epoch+1}/{epochs}, Loss: Invalid (NaN/Inf)")




# Optional: Plot losses
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 3))
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()




print("Step 4.1: Preparing input image and prompt for generation...")


test_image_path = image_paths["green"]
test_prompt_text = "Describe this image: "

# --- Process Image ---
try:
    test_img = Image.open(test_image_path).convert('RGB')
    test_img_tensor = image_transforms(test_img).unsqueeze(0).to(device)
    with torch.no_grad(): # No gradients needed for feature extraction
        test_img_features_raw = vision_model(test_img_tensor) # (1, vision_feature_dim)
    # Project features using the TRAINED projection layer
    vision_projection_layer.eval() # Ensure projection layer is in eval mode
    with torch.no_grad():
        test_img_features_projected = vision_projection_layer(test_img_features_raw) # (1, d_model)
    print(f"  Processed image: '{os.path.basename(test_image_path)}'")
    print(f"  Projected image features shape: {test_img_features_projected.shape}")
except FileNotFoundError:
    print(f"Error: Test image not found at {test_image_path}. Cannot generate.")
    # Handle error appropriately, maybe exit or skip generation
    test_img_features_projected = None # Indicate error

# --- Process Prompt ---
if test_img_features_projected is not None:
    img_id = char_to_int[img_token]
    prompt_ids = [char_to_int[ch] for ch in test_prompt_text]
    # Initial sequence IDs: [<IMG>, prompt tokens]
    initial_context_ids = torch.tensor([[img_id] * num_img_tokens + prompt_ids], dtype=torch.long, device=device) # Shape: (1, 1 + len(prompt))
    print(f"  Tokenized prompt: '{test_prompt_text}' -> {initial_context_ids.tolist()}")
else:
     initial_context_ids = None

# --- Generation Parameters ---
max_new_tokens = 50 # Max characters to generate for the response
eos_token_id = char_to_int[eos_token]
print(f"  Set max new tokens to generate: {max_new_tokens}")



# --- Set Model to Evaluation Mode ---
token_embedding_table.eval()
# vision_projection_layer is already eval
for i in range(n_layers):
    layer_norms_1[i].eval()
    mha_qkv_linears[i].eval()
    mha_output_linears[i].eval()
    layer_norms_2[i].eval()
    ffn_linear_1[i].eval()
    ffn_linear_2[i].eval()
final_layer_norm.eval()
output_linear_layer.eval()

# --- Generation ---
generated_sequence_ids = initial_context_ids # Start with image + prompt
if generated_sequence_ids is not None:
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # --- Prepare Input ---
            current_ids_context = generated_sequence_ids[:, -block_size:] # Ensure context fits block_size
            B_gen, T_gen = current_ids_context.shape
            C_gen = d_model

            # Get embeddings for current context IDs
            current_token_embeddings = token_embedding_table(current_ids_context) # (B_gen, T_gen, C_gen)

            # Inject image features (only needed if <IMG> is within the current context window)
            # Find the position of the <IMG> token(s) in the current context
            img_token_pos = -1
            if img_id in current_ids_context[0].tolist():
                 # Simple check assuming only one <IMG> at the start of the original sequence
                 if current_ids_context[0, 0] == img_id:
                      img_token_pos = 0

            gen_combined_embeddings = current_token_embeddings
            if img_token_pos != -1: # Check if the image token is present in the window
                 # Inject pre-calculated projected features for the image token(s)
                 # Assume only one image token at pos 0 for simplicity
                 gen_combined_embeddings[:, img_token_pos:(img_token_pos + num_img_tokens), :] = test_img_features_projected # (1, C) broadcasted/sliced


            # Add Positional Encoding (sliced for current context length T_gen)
            pos_enc_slice_gen = positional_encoding[:, :T_gen, :]
            x_gen = gen_combined_embeddings + pos_enc_slice_gen

            # --- Transformer Blocks ---
            # Create causal mask for current length T_gen
            gen_causal_mask = causal_mask[:,:,:T_gen,:T_gen] # (1, 1, T_gen, T_gen)
            # No padding mask needed here as we generate one token at a time / handle context length

            for i in range(n_layers):
                x_input_block_gen = x_gen
                # Pre-LN MHA
                x_ln1_gen = layer_norms_1[i](x_input_block_gen)
                qkv_gen = mha_qkv_linears[i](x_ln1_gen)
                qkv_gen = qkv_gen.view(B_gen, T_gen, n_heads, 3 * d_k).permute(0, 2, 1, 3)
                q_gen, k_gen, v_gen = qkv_gen.chunk(3, dim=-1)
                attn_scores_gen = (q_gen @ k_gen.transpose(-2, -1)) * (d_k ** -0.5)
                attn_scores_masked_gen = attn_scores_gen.masked_fill(gen_causal_mask == 0, float('-inf'))
                attention_weights_gen = F.softmax(attn_scores_masked_gen, dim=-1)
                attention_weights_gen = torch.nan_to_num(attention_weights_gen)
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

            # --- Final Layers ---
            final_norm_output_gen = final_layer_norm(x_gen)
            logits_gen = output_linear_layer(final_norm_output_gen) # (B_gen, T_gen, vocab_size)

            # --- Get Logits for Last Token ---
            logits_last_token = logits_gen[:, -1, :] # (B_gen, vocab_size)

            # --- Sample Next Token ---
            probs = F.softmax(logits_last_token, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1) # (B_gen, 1)

            # --- Append and Check for EOS ---
            generated_sequence_ids = torch.cat((generated_sequence_ids, next_token_id), dim=1)

            if next_token_id.item() == eos_token_id:
                print("  <EOS> token generated. Stopping.")
                break
        else: # Loop finished without hitting EOS
             print(f"  Reached max generation length ({max_new_tokens}). Stopping.")

else:
    print("Generation skipped due to error in preparing input.")

print("--- Generation Loop Finished ---")



if generated_sequence_ids is not None:
    # Extract the generated IDs (excluding the initial <IMG> token if desired)
    final_ids_list = generated_sequence_ids[0].tolist()

    # Decode the full sequence
    decoded_text = ""
    for id_val in final_ids_list:
        # Avoid trying to decode potential ignore_index if it slipped through
        if id_val in int_to_char:
            decoded_text += int_to_char[id_val]
        else:
            decoded_text += f"[UNK:{id_val}]" # Handle unknown IDs
    # Displaying the prompt and response part
    # Find start of response (first token after prompt)
    response_start_index = num_img_tokens + len(test_prompt_text)
    print(f"Prompt: {test_prompt_text}")
    print(f"Generated Response: {decoded_text[response_start_index:]}")
    # print(f"Full Decoded Sequence: {decoded_text}") # Optional: print everything
else:
    print("Decoding skipped.")





import os

# Create the directory if it doesn't exist
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'multimodal_model.pt')

# Create a dictionary with all model components and configurations
multimodal_state_dict = {
    # Configuration
    'config': {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'block_size': block_size,
        'num_img_tokens': num_img_tokens,
        'vision_feature_dim': vision_feature_dim
    },
    # Tokenizer
    'tokenizer': {
        'char_to_int': char_to_int,
        'int_to_char': int_to_char
    },
    # Model weights
    'token_embedding_table': token_embedding_table.state_dict(),
    'vision_projection_layer': vision_projection_layer.state_dict(),
    'positional_encoding': positional_encoding,
    'layer_norms_1': [ln.state_dict() for ln in layer_norms_1],
    'mha_qkv_linears': [l.state_dict() for l in mha_qkv_linears],
    'mha_output_linears': [l.state_dict() for l in mha_output_linears],
    'layer_norms_2': [ln.state_dict() for ln in layer_norms_2],
    'ffn_linear_1': [l.state_dict() for l in ffn_linear_1],
    'ffn_linear_2': [l.state_dict() for l in ffn_linear_2],
    'final_layer_norm': final_layer_norm.state_dict(),
    'output_linear_layer': output_linear_layer.state_dict()
}

# Save to file
torch.save(multimodal_state_dict, save_path)
print(f"Multi-modal model saved to {save_path}")



# Load the saved model state dictionary
model_load_path = 'saved_models/multimodal_model.pt'
loaded_state_dict = torch.load(model_load_path, map_location=device)
print(f"Loaded state dictionary from '{model_load_path}'.")

# Extract configuration and tokenizer
config = loaded_state_dict['config']
vocab_size = config['vocab_size']
d_model = config['d_model']
n_heads = config['n_heads']
n_layers = config['n_layers']
d_ff = config['d_ff']
block_size = config['block_size']
num_img_tokens = config['num_img_tokens']
vision_feature_dim = config['vision_feature_dim']
d_k = d_model // n_heads

char_to_int = loaded_state_dict['tokenizer']['char_to_int']
int_to_char = loaded_state_dict['tokenizer']['int_to_char']

# Recreate causal mask
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)

# Rebuild model components
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)
token_embedding_table.load_state_dict(loaded_state_dict['token_embedding_table'])

vision_projection_layer = nn.Linear(vision_feature_dim, d_model).to(device)
vision_projection_layer.load_state_dict(loaded_state_dict['vision_projection_layer'])

positional_encoding = loaded_state_dict['positional_encoding'].to(device)

# Initialize transformer layers
layer_norms_1 = []
mha_qkv_linears = []
mha_output_linears = []
layer_norms_2 = []
ffn_linear_1 = []
ffn_linear_2 = []

for i in range(n_layers):
    # For each layer, create components and load their state dictionaries
    ln1 = nn.LayerNorm(d_model).to(device)
    ln1.load_state_dict(loaded_state_dict['layer_norms_1'][i])
    layer_norms_1.append(ln1)
    
    # For the QKV linear layer, we need to match the bias parameter
    qkv_dict = loaded_state_dict['mha_qkv_linears'][i]
    has_qkv_bias = 'bias' in qkv_dict
    qkv = nn.Linear(d_model, 3 * d_model, bias=has_qkv_bias).to(device)
    qkv.load_state_dict(qkv_dict)
    mha_qkv_linears.append(qkv)
    
    # Similar approach for other linear layers
    out_dict = loaded_state_dict['mha_output_linears'][i]
    has_out_bias = 'bias' in out_dict
    out = nn.Linear(d_model, d_model, bias=has_out_bias).to(device)
    out.load_state_dict(out_dict)
    mha_output_linears.append(out)
    
    ln2 = nn.LayerNorm(d_model).to(device)
    ln2.load_state_dict(loaded_state_dict['layer_norms_2'][i])
    layer_norms_2.append(ln2)
    
    ff1_dict = loaded_state_dict['ffn_linear_1'][i]
    has_ff1_bias = 'bias' in ff1_dict
    ff1 = nn.Linear(d_model, d_ff, bias=has_ff1_bias).to(device)
    ff1.load_state_dict(ff1_dict)
    ffn_linear_1.append(ff1)
    
    ff2_dict = loaded_state_dict['ffn_linear_2'][i]
    has_ff2_bias = 'bias' in ff2_dict
    ff2 = nn.Linear(d_ff, d_model, bias=has_ff2_bias).to(device)
    ff2.load_state_dict(ff2_dict)
    ffn_linear_2.append(ff2)

# Final layer norm and output projection
final_layer_norm = nn.LayerNorm(d_model).to(device)
final_layer_norm.load_state_dict(loaded_state_dict['final_layer_norm'])

output_dict = loaded_state_dict['output_linear_layer']
has_output_bias = 'bias' in output_dict
output_linear_layer = nn.Linear(d_model, vocab_size, bias=has_output_bias).to(device)
output_linear_layer.load_state_dict(output_dict)

print("Multi-modal model components loaded successfully.")




def generate_with_image(image_path, prompt, max_new_tokens=50):
    """Generate text response for an image and prompt"""
    # Set everything to evaluation mode
    token_embedding_table.eval()
    vision_projection_layer.eval()
    for i in range(n_layers):
        layer_norms_1[i].eval()
        mha_qkv_linears[i].eval()
        mha_output_linears[i].eval()
        layer_norms_2[i].eval()
        ffn_linear_1[i].eval()
        ffn_linear_2[i].eval()
    final_layer_norm.eval()
    output_linear_layer.eval()
    
    # Process image
    image = Image.open(image_path).convert('RGB')
    img_tensor = image_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract and project image features
        img_features_raw = vision_model(img_tensor)
        img_features_projected = vision_projection_layer(img_features_raw)
        
        # Tokenize prompt and prepare initial sequence
        img_id = char_to_int[img_token]
        prompt_ids = [char_to_int[ch] for ch in prompt]
        context_ids = torch.tensor([[img_id] + prompt_ids], dtype=torch.long, device=device)
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Use only the last block_size tokens if context gets too long
            context_ids = context_ids[:, -block_size:]
            

