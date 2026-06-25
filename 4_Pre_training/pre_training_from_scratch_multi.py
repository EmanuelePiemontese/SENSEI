import os
import sys
import random
import yaml
import time
import torch
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "SENSEI_config.yaml")

# ---------------------------
# SEED
# ---------------------------
def set_seed(seed):
    """Set the random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# DISTRIBUTED SETUP
# ---------------------------
def setup_distributed():
    """Initialize the process group for Distributed Data Parallel (DDP)."""
    is_distributed = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ["WORLD_SIZE"]) > 1
    )

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend = "nccl",
            device_id = device,  # Eliminates the "guessing device" warning
        )
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return is_distributed, local_rank, device


# ---------------------------
# CONFIGURATION
# ---------------------------
def load_config(path = DEFAULT_CONFIG_PATH):
    """Load the YAML configuration file."""
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------
# DATASET
#
# Reads from pre-calculated numpy memmap files.
# Receives an array of arbitrary indices (already shuffled and split in main).
# Sequential block access -> Fast forward reading on Lustre file systems.
# ---------------------------
class XESDatasetFast(Dataset):
    def __init__(self, ids_path: str, mask_path: str, indices: np.ndarray, shuffle_chunks: bool = False, seed: int = 42):
        """
        ids_path, mask_path : .npy files produced by the dataset conversion script
        indices             : Numpy array of partition indices (globally shuffled and split in main)
        shuffle_chunks      : If True, shuffles indices in 50k blocks to maintain locality on Lustre during training
        seed                : Seed for reproducible shuffling
        """
        # mmap_mode = "r" -> File shared across workers, zero RAM copy overhead
        self.input_ids = np.load(ids_path, mmap_mode = "r")
        self.attention_mask = np.load(mask_path, mmap_mode = "r")

        if shuffle_chunks:
            # Block shuffling: each chunk is sequential on Lustre,
            # but the chunks are shuffled among themselves -> Good compromise
            # between randomness and read speed on network filesystems.
            chunk_size = 50000
            chunks = [
                indices[i : i + chunk_size]
                for i in range(0, len(indices), chunk_size)
            ]
            rng = np.random.RandomState(seed)
            rng.shuffle(chunks)
            self.indices = np.concatenate(chunks)
        else:
            self.indices = indices
            
        shuffle_status = "chunk" if shuffle_chunks else "no"
        print(f"[Dataset] {len(self.indices):,} samples, shuffle = {shuffle_status}", flush = True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        return {
            "input_ids": torch.from_numpy(self.input_ids[real_idx].astype(np.int64)),
            "attention_mask": torch.from_numpy(self.attention_mask[real_idx].astype(np.int64)),
        }


# ---------------------------
# MAIN
# ---------------------------
def main():
    config = load_config()
    p = config["paths"]
    m = config["model"]
    t = config["training"]

    seed_val = t["seed"]
    set_seed(seed_val)

    is_distributed, local_rank, device = setup_distributed()

    if local_rank == 0:
        print(f"[INFO] Python:  {sys.version}", flush = True)
        print(f"[INFO] PyTorch: {torch.__version__}", flush = True)
        print(f"[INFO] CUDA:    {torch.version.cuda}", flush = True)
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush = True)
        print(f"\n[INFO] Distributed: {is_distributed}", flush = True)
        print(f"[INFO] Device:      {device}", flush = True)

    # ---------------------------
    # TOKENIZER & PATH RESOLUTION
    # ---------------------------
    # We resolve the absolute root directory of the project here
    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    
    tokenizer_dir = os.path.join(root_dir, p["tokenizer_dir"])
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

    # ---------------------------
    # DATASET
    # ---------------------------
    dataset_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "dataset_pre_training"))
    
    ids_path = os.path.join(dataset_dir, "input_ids.npy")
    mask_path = os.path.join(dataset_dir, "attention_mask.npy")

    if not (os.path.exists(ids_path) and os.path.exists(mask_path)):
        raise FileNotFoundError(
            f"[ERROR] Memmap files not found:\n  {ids_path}\n  {mask_path}\n"
            "Please run the dataset conversion script first."
        )

    # Read only the shape without loading the data into RAM
    probe_data = np.load(ids_path, mmap_mode = "r")
    n_total = len(probe_data)
    del probe_data

    # Global shuffle of indices BEFORE splitting
    # -> Train and Val contain uniformly distributed samples
    # -> Avoids distortions due to the original order of the JSONL file
    rng = np.random.RandomState(seed_val)
    shuffled_idx = rng.permutation(n_total)

    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:]

    if local_rank == 0:
        print(f"[Dataset] Total: {n_total:,}", flush = True)
        print(f"[Dataset] Train: {n_train:,}", flush = True)
        print(f"[Dataset] Val:   {n_val:,}", flush = True)

    # shuffle_chunks = True in train: shuffles in 50k blocks
    # to vary the order at each epoch while maintaining locality on Lustre
    train_dataset = XESDatasetFast(
        ids_path, 
        mask_path,
        indices = train_idx,
        shuffle_chunks = True,  
        seed = seed_val,
    )
    val_dataset = XESDatasetFast(
        ids_path, 
        mask_path,
        indices = val_idx,
        shuffle_chunks = False, 
        seed = seed_val,
    )

    # ---------------------------
    # COLLATOR
    # DataCollatorForLanguageModeling generates labels with -100
    # on non-masked tokens — do not pass input_ids as labels!
    # ---------------------------
    mlm_prob = t["mlm_probability"]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = True,
        mlm_probability = mlm_prob,
    )

    # ---------------------------
    # SAMPLERS
    # shuffle = False: Shuffling is already managed by XESDatasetFast
    # in sequential blocks -> Lustre does not perform random seeks
    # ---------------------------
    train_sampler = (
        DistributedSampler(train_dataset, shuffle = False)
        if is_distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle = False)
        if is_distributed else None
    )

    # ---------------------------
    # DATALOADERS
    # ---------------------------
    num_workers = t.get("num_workers", 4)
    batch_size_val = t["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size_val,
        sampler = train_sampler,
        shuffle = False,  # Managed by XESDatasetFast
        collate_fn = data_collator,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 2,
        persistent_workers = (num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size_val,
        sampler = val_sampler,
        shuffle = False,
        collate_fn = data_collator,
        num_workers = num_workers,
        pin_memory = True,
        prefetch_factor = 2,
        persistent_workers = (num_workers > 0),
    )

    if local_rank == 0:
        print(f"[Loader] Batch Size = {batch_size_val}  Num Workers = {num_workers}", flush = True)
        print(f"[Loader] Train batches: {len(train_loader):,}", flush = True)
        print(f"[Loader] Val batches:   {len(val_loader):,}", flush = True)

    # ---------------------------
    # MODEL (BERT small configuration) --> SENSEI
    # ---------------------------
    config_bert = BertConfig(
        vocab_size = len(tokenizer),
        hidden_size = m["hidden_size"],
        num_hidden_layers = m["num_hidden_layers"],
        num_attention_heads = m["num_attention_heads"],
        intermediate_size = m["intermediate_size"],
        max_position_embeddings = m["max_len"],
    )

    model = BertForMaskedLM(config_bert).to(device)

    if local_rank == 0:
        n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        print(f"[Model] Parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush = True)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids = [local_rank],
            output_device = local_rank,
            find_unused_parameters = False,
        )

    # ---------------------------
    # MIXED PRECISION
    # bfloat16: Natively supported by Ampere/Hopper (H100), more stable than fp16,
    # does not require GradScaler.
    # ---------------------------
    use_amp = t.get("use_amp", True)
    amp_dtype = torch.bfloat16 if use_amp else None

    if local_rank == 0 and use_amp:
        print("[AMP] bfloat16 enabled", flush = True)

    # ---------------------------
    # OPTIMIZER + SCHEDULER
    # ---------------------------
    lr_val = float(t["learning_rate"])
    decay_val = t["weight_decay"]
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr_val,
        weight_decay = decay_val,
        fused = True,  # Fused AdamW kernel — significantly faster on CUDA
    )

    t_epochs = t["epochs"]
    total_steps = len(train_loader) * t_epochs
    warmup_val = t["warmup_steps"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = warmup_val,
        num_training_steps = total_steps,
    )

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    best_val = float("inf")
    patience_counter = 0
    max_patience = t["patience"]

    for epoch in range(t_epochs):

        if is_distributed:
            train_sampler.set_epoch(epoch)

        if local_rank == 0:
            separator_line = "=" * 50
            print(f"\n{separator_line}", flush = True)
            print(f"Epoch {epoch + 1}/{t_epochs}", flush = True)

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for batch in tqdm(train_loader, disable = (local_rank != 0), desc = "Training"):

            input_ids = batch["input_ids"].to(device, non_blocking = True)
            attention_mask = batch["attention_mask"].to(device, non_blocking = True)
            labels = batch["labels"].to(device, non_blocking = True)

            optimizer.zero_grad(set_to_none = True)

            with torch.autocast(device_type = "cuda", dtype = amp_dtype, enabled = use_amp):
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                )
                loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if is_distributed:
            t_tensor = torch.tensor(train_loss, device = device)
            dist.all_reduce(t_tensor, op = dist.ReduceOp.AVG)
            train_loss = t_tensor.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, disable = (local_rank != 0), desc = "Validation"):

                input_ids = batch["input_ids"].to(device, non_blocking = True)
                attention_mask = batch["attention_mask"].to(device, non_blocking = True)
                labels = batch["labels"].to(device, non_blocking = True)

                with torch.autocast(device_type = "cuda", dtype = amp_dtype, enabled = use_amp):
                    outputs = model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels,
                    )

                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)

        if is_distributed:
            v_tensor = torch.tensor(val_loss, device = device)
            dist.all_reduce(v_tensor, op = dist.ReduceOp.AVG)
            val_loss = v_tensor.item()

        epoch_time = time.time() - epoch_start

        if local_rank == 0:
            print(f"Train loss: {train_loss:.4f}", flush = True)
            print(f"Val loss:   {val_loss:.4f}", flush = True)
            print(f"Time:       {epoch_time / 60:.1f} min", flush = True)

        # ---- CHECKPOINT (rank 0 only) ----
        if local_rank == 0:
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0

                unwrapped = model.module if is_distributed else model
                
                # Resolving the absolute path for checkpoints
                checkpoint_base = os.path.join(root_dir, p["checkpoint_path"])
                checkpoint_dir = os.path.join(checkpoint_base, f"best_epoch_{epoch + 1:03d}")

                os.makedirs(checkpoint_dir, exist_ok = True)

                unwrapped.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                print(f"[SUCCESS] Checkpoint saved: {checkpoint_dir} (val_loss = {best_val:.4f})", flush = True)
            else:
                patience_counter += 1
                print(f"[INFO] Patience: {patience_counter}/{max_patience}", flush = True)

        # Synchronize patience_counter across ranks -> prevents early stopping deadlocks
        if is_distributed:
            pc_tensor = torch.tensor(patience_counter, device = device)
            dist.broadcast(pc_tensor, src = 0)
            patience_counter = int(pc_tensor.item())
            dist.barrier()

        if patience_counter >= max_patience:
            if local_rank == 0:
                print("[INFO] Early stopping triggered", flush = True)
            break

    # ---------------------------
    # FINAL SAVE
    # ---------------------------
    if local_rank == 0:
        # Resolving the absolute path for the final model output
        final_output_dir = os.path.join(root_dir, p["output_dir"])
        os.makedirs(final_output_dir, exist_ok = True)
        
        unwrapped = model.module if is_distributed else model
        unwrapped.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        print("\n[SUCCESS] Training successfully completed!", flush = True)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()