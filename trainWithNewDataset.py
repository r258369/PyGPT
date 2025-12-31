import os
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Set environment variables for CUDA debugging (optional)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Define paths for the new dataset
paths = ["/content/drive/MyDrive/new_dataset.txt"]  # Replace with your new dataset path

# Step 1: Train a new tokenizer for the new dataset
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=paths,
    vocab_size=52_000,  # Same vocab size as before, adjust if needed
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.save_model("/content/drive/MyDrive/tokenizer")

# Load the new tokenizer as a GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/MyDrive/tokenizer")
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})

model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/GPyT")

# Step 3: Resize the model’s token embeddings to match the new tokenizer’s vocabulary
model.resize_token_embeddings(len(tokenizer))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Load and prepare the new dataset
dataset = load_dataset("text", data_files=paths)

# Define the encoding function using the new tokenizer
def encode(lines):
    return tokenizer(
        lines['text'],
        add_special_tokens=True,
        truncation=True,
        max_length=512,  # Adjust max_length if needed
        padding="max_length",
    )

# Apply the encoding transformation
dataset.set_transform(encode)
dataset = dataset['train']

# Step 5: Define data collator with the new tokenizer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Set training arguments for the new training
new_training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/GPyT_new",  # New output directory
    overwrite_output_dir=True,
    num_train_epochs=1,  # Adjust as needed
    per_device_train_batch_size=10,  # Adjust as needed
    save_steps=5_000,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False,
)

# Step 7: Initialize Trainer with the updated model and new tokenizer
trainer = Trainer(
    model=model,
    args=new_training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Step 8: Start training on the new dataset
trainer.train()

# Step 9: Save the updated model and new tokenizer
trainer.save_model("/content/drive/MyDrive/GPyT_new")
tokenizer.save_pretrained("/content/drive/MyDrive/GPyT_new")  # Explicitly save the new tokenizer