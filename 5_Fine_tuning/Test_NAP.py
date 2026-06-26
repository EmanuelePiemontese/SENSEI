import os
import gc
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertModel, BertConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Dynamic paths aligned with the project structure
PATH_PICKLE_DIR = os.path.join(ROOT_DIR, "results", "Pre_processing_test", "NAP")
PATH_TOKENIZER = os.path.join(ROOT_DIR, "model", "tokenizer_abstract_symbol_mapping")
SENSEI_PRETRAINED_PATH = os.path.join(ROOT_DIR, "model", "SENSEI_pre_trained")
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, "model", "SENSEI_fine_tuned_NAP")

# New folder for checkpoints and results
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "model", "checkpoints_NAP")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "Fine_tuning_benchmark")

# --- SEED SETTING ---
def set_seed(seed):
    """Ensure fully reproducible results across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- CONFIGURATION ---
t = {
    "epochs": 15,
    "patience": 3,
    "seed": 42,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_len": 256
}

set_seed(t["seed"])

models_to_test = {
    "Google_BERT": {
        "path": "google/bert_uncased_L-4_H-512_A-8",
        "use_custom_tokenizer": False,
    },
    "SENSEI_Process_BERT": {
        "path": SENSEI_PRETRAINED_PATH,
        "use_custom_tokenizer": True,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_train_val(X, y, val_ratio = 0.2, random_state = 42):
    """Stratified train/val split."""
    return train_test_split(
        X, y,
        test_size = val_ratio,
        random_state = random_state,
        shuffle = True,
        stratify = y # Stratify added to ensure class balance
    )

# --- MODEL AND DATASET CLASSES ---
class BertForMeanPoolingClassification(nn.Module):
    """
    Custom BERT model for classification using mean pooling over token embeddings.
    """
    def __init__(self, model_path, num_labels, special_token_ids, is_scratch = False):
        """
        Initialize the model with a BERT backbone and a classification head.
        """
        super(BertForMeanPoolingClassification, self).__init__()
        config = BertConfig.from_pretrained(model_path)

        if is_scratch:
            self.bert = BertModel(config)
        else:
            self.bert = BertModel.from_pretrained(model_path)

        self.num_labels = num_labels
        self.cls_id = special_token_ids["cls"]
        self.sep_id = special_token_ids["sep"]
        self.pad_id = special_token_ids["pad"]

        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask = None, labels = None):
        """
        Forward pass through the model.
        """
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        last_hidden_state = outputs.last_hidden_state

        mask = (
            (input_ids != self.cls_id) &
            (input_ids != self.sep_id) &
            (input_ids != self.pad_id)
        )
        mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min = 1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


class NAPDataset(Dataset):
    """
    Custom Dataset class for the NAP dataset, handling both abstractive and original text formats.
    """
    def __init__(self, texts, labels, tok, is_split, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.is_split = is_split
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # If is_split is True (Custom Model), Hugging Face REQUIRES a list.
        # Since we have strings in the .pkl file, we divide them using .split()
        if self.is_split and isinstance(text, str):
            text = text.split()
        # If is_split is False (Google BERT), Hugging Face REQUIRES a string.
        elif not self.is_split and isinstance(text, list):
            text = " ".join(text)

        encodings = self.tok(
            text,
            is_split_into_words = self.is_split,
            truncation = True,
            padding = "max_length",
            max_length = self.max_len,
            return_tensors = "pt"
        )

        item = {key: val.squeeze(0).contiguous() for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype = torch.long)
        return item

# --- MAIN TRAINING AND TEST LOOP ---
def main():
    final_comparison_results = []

    if not os.path.exists(PATH_PICKLE_DIR):
        raise FileNotFoundError(f"[ERROR] NAP dataset directory not found at: {PATH_PICKLE_DIR}")

    pkl_files = [
        os.path.join(PATH_PICKLE_DIR, f)
        for f in os.listdir(PATH_PICKLE_DIR)
        if f.endswith(".pkl")
    ]

    for pkl_path in pkl_files:
        log_name = os.path.splitext(os.path.basename(pkl_path))[0]
        separator_line = "=" * 80
        print(f"\n{separator_line}")
        print(f"[INFO] LOADING DATASET: {log_name}")
        print(separator_line)

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Create the Universal Label Encoding Vocabulary
        # We find all possible next activities by joining train and test (both abs and txt)
        unique_labels = sorted(list(set(data["y_train_abs"] + data["y_test_abs"])))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        num_labels = len(unique_labels)

        for model_name, cfg in models_to_test.items():
            print(f"\n[INFO] >>> Model: {model_name}")

            model_path = cfg["path"]
            use_custom = cfg["use_custom_tokenizer"]

            # 1. TOKENIZER AND DATA SETUP
            if use_custom:
                X_train_full = data["X_train_abs"]
                X_test = data["X_test_abs"]
                
                # -> LOAD ABS LABELS
                y_train_full = [label2id[label] for label in data["y_train_abs"]]
                y_test = [label2id[label] for label in data["y_test_abs"]]
                
                tokenizer = BertTokenizerFast.from_pretrained(PATH_TOKENIZER)
                
                # ---> VOCABULARY KEY CORRECTION HERE <---
                tokenizer.add_tokens(list(data["activity_map"].values()))
                
                is_split = True
                print("  -> Tokenizer: Custom | Text: Abstractive (abs)")
            else:
                X_train_full = data["X_train_txt"]
                X_test = data["X_test_txt"]
                
                # -> LOAD TXT LABELS
                # Create a specific label_encoder for txt
                unique_labels_txt = sorted(list(set(data["y_train_txt"] + data["y_test_txt"])))
                label2id_txt = {label: idx for idx, label in enumerate(unique_labels_txt)}

                y_train_full = [label2id_txt[label] for label in data["y_train_txt"]]
                y_test = [label2id_txt[label] for label in data["y_test_txt"]]
                
                tokenizer = BertTokenizerFast.from_pretrained(model_path)
                is_split = False
                print("  -> Tokenizer: Original BERT | Text: Original (txt)")

            # Stratified split train -> train + val
            X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)
            print(
                f"  Classes: {num_labels} | "
                f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}"
            )

            sp_ids = {
                "cls": tokenizer.cls_token_id,
                "sep": tokenizer.sep_token_id,
                "pad": tokenizer.pad_token_id,
            }

            # 2. CHECKPOINT AND OUTPUT PATH
            safe_model_name = model_name.replace("/", "_").replace(" ", "_")
            
            # Directory creation
            os.makedirs(CHECKPOINT_DIR, exist_ok = True)
            os.makedirs(OUTPUT_BASE_DIR, exist_ok = True)
            
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{log_name}_{safe_model_name}.pt")
            output_dir = os.path.join(OUTPUT_BASE_DIR, f"{log_name}_{safe_model_name}")
            os.makedirs(output_dir, exist_ok = True)

            # 3. MODEL AND DATALOADERS
            model_ft = BertForMeanPoolingClassification(
                model_path, 
                num_labels, 
                sp_ids,
                is_scratch = ("scratch" in model_name.lower())
            )
            model_ft.bert.resize_token_embeddings(len(tokenizer))
            model_ft.to(device)

            # ---> MULTI-GPU UNLOCK <---
            if torch.cuda.device_count() > 1:
                print(f"[INFO] Hardware unlocked: Using {torch.cuda.device_count()} GPUs in parallel!")
                model_ft = nn.DataParallel(model_ft)

            # ---> MULTI-CORE CPU UNLOCK & INCREASED BATCHES <---
            train_loader = DataLoader(
                NAPDataset(X_train, y_train, tokenizer, is_split, t["max_len"]),
                batch_size = 256, 
                shuffle = True, 
                pin_memory = True, 
                num_workers = 2
            )
            val_loader = DataLoader(
                NAPDataset(X_val, y_val, tokenizer, is_split, t["max_len"]),
                batch_size = 256, 
                shuffle = False, 
                pin_memory = True, 
                num_workers = 2
            )
            test_loader = DataLoader(
                NAPDataset(X_test, y_test, tokenizer, is_split, t["max_len"]),
                batch_size = 128, 
                shuffle = False, 
                pin_memory = True, 
                num_workers = 2
            )

            optimizer = torch.optim.AdamW(
                model_ft.parameters(),
                lr = t["learning_rate"],
                weight_decay = t["weight_decay"]
            )

            num_training_steps = len(train_loader) * t["epochs"]
            warmup_steps = int(0.1 * num_training_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = warmup_steps,
                num_training_steps = num_training_steps
            )

            # Support variables to format f-strings safely
            total_epochs = t["epochs"]
            max_patience = t["patience"]

            # 4. TRAINING LOOP WITH EARLY STOPPING ON VAL
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(total_epochs):
                model_ft.train()
                train_loss = 0
                train_bar = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{total_epochs} Training")

                for batch in train_bar:
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model_ft(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                    
                    # Loss becomes a tensor with multiple values if we use DataParallel, so we take the mean
                    loss = outputs["loss"].mean() if torch.cuda.device_count() > 1 else outputs["loss"]
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_ft.parameters(), max_norm = 1.0)
                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    train_bar.set_postfix(loss = f"{loss.item():.4f}")

                # Validation on val set
                model_ft.eval()
                val_loss = 0
                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc = "Validation")
                    for batch in val_bar:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
                        
                        outputs = model_ft(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                        v_loss = outputs["loss"].mean() if torch.cuda.device_count() > 1 else outputs["loss"]
                        val_loss += v_loss.item()

                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                print(f"[INFO] Loss -> Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

                # ---> PURE MODEL EXTRACTION FOR SAVING <---
                model_to_save = model_ft.module if hasattr(model_ft, "module") else model_ft

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model_to_save.state_dict(), checkpoint_path)
                    print(f"[SUCCESS] Checkpoint saved: {checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"[INFO] No improvement ({patience_counter}/{max_patience})")
                    if patience_counter >= max_patience:
                        print("[INFO] Early stopping triggered.")
                        break

            # 5. FINAL EVALUATION ON TEST SET
            # ---> LOADING ON THE PURE MODEL <---
            model_to_save = model_ft.module if hasattr(model_ft, "module") else model_ft
            model_to_save.load_state_dict(torch.load(checkpoint_path, map_location = device, weights_only = True))
            model_ft.eval()
            
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model_ft(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                    preds = torch.argmax(outputs["logits"], dim = -1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average = "macro")
            print(f"\n[TEST] Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")

            final_comparison_results.append({
                "Log": log_name,
                "Model": model_name,
                "Accuracy": acc,
                "F1_Macro": f1,
            })

            # 6. MODEL SAVING
            print(f"[INFO] Saving model in: {output_dir}")
            state_dict = {
                k: v.contiguous() for k, v in model_to_save.state_dict().items()
            }
            torch.save(state_dict, os.path.join(output_dir, "model_state_dict.pt"))
            tokenizer.save_pretrained(output_dir)
            model_to_save.bert.config.save_pretrained(output_dir)
            print(f"[SUCCESS] Final model saved in: {output_dir}")

            # VRAM Cleanup
            del model_ft, train_loader, val_loader, test_loader, optimizer, scheduler, tokenizer, model_to_save
            gc.collect()
            torch.cuda.empty_cache()

        # RAM Cleanup between pkl files
        del data, X_train_full, X_test, y_train_full, y_test
        gc.collect()

    # 7. FINAL REPORT
    os.makedirs(RESULTS_DIR, exist_ok = True)
    results_df = pd.DataFrame(final_comparison_results)
    
    csv_output_path = os.path.join(RESULTS_DIR, "risultati_nap_benchmark.csv")
    results_df.to_csv(csv_output_path, index = False)
    
    separator_line = "=" * 80
    print(f"\n{separator_line}")
    print(f"[SUCCESS] Benchmark completed. Results saved in '{csv_output_path}'.")

    try:
        display(results_df)
    except NameError:
        print(results_df.to_string(index = False))

if __name__ == "__main__":
    main()