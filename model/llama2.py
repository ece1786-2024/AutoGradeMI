import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    LlamaTokenizer, 
    LlamaForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import pandas as pd
import numpy as np
import sentencepiece
from sklearn.metrics import accuracy_score, classification_report
import bitsandbytes as bnb
from torch.cuda.amp import autocast


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        

        text = f"{item['prompt']}\n{item['essay']}"
        

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def load_model_and_tokenizer(model_name="NousResearch/Llama-2-7b-hf", num_labels=12):
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = LlamaForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        num_labels=num_labels,
        quantization_config=quantization_config
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ]
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataloaders(data_path, tokenizer, batch_size=2):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    

    train_df = df.sample(n=2000, random_state=42)
    temp_df = df[~df.index.isin(train_df.index)]
    val_df = temp_df.sample(n=500, random_state=42)
    test_df = temp_df[~temp_df.index.isin(val_df.index)].sample(n=500, random_state=42)


    train_dataset = CustomDataset(train_df.to_dict('records'), tokenizer)
    val_dataset = CustomDataset(val_df.to_dict('records'), tokenizer)
    test_dataset = CustomDataset(test_df.to_dict('records'), tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss


        loss.backward()
        

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(eval_loader), accuracy

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    BATCH_SIZE = 2
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    
    model, tokenizer = load_model_and_tokenizer()

    train_loader, val_loader, test_loader = prepare_dataloaders(
        '../data/dataset/processed/clean_data_gpt2.csv',
        tokenizer,
        BATCH_SIZE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.1,
        total_iters=len(train_loader) * EPOCHS
    )

    print("Starting training...")
    best_val_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
  
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
 
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained("./best_model")
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
    

    print("\nLoading best model for testing...")
    model = PeftModel.from_pretrained(model, "./best_model")
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":

    torch.cuda.set_per_process_memory_fraction(0.95)
    main()