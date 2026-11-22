import argparse
import csv
import gc
import json
import os
import pickle
import random
import re
import resource
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

UNK = "[UNK]"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODES = {
    "scratch": dict(
        pretrained=False,
        freeze=False,
        grid=[dict(lr_enc=lr, lr_head=lr) for lr in ([1e-4])],
    ),
    "finetune": dict(
        pretrained=True,
        freeze=False,
        grid=[dict(lr_enc=lr, lr_head=1e-3) for lr in ([1e-5])],
    ),
    "frozen": dict(
        pretrained=True,
        freeze=True,
        grid=[dict(lr_enc=0.0, lr_head=lr) for lr in ([1e-3])],
    ),
    "frozen_random": dict(
        pretrained=False,
        freeze=True,
        grid=[dict(lr_enc=0.0, lr_head=lr) for lr in ([1e-3])],
    ),
}

RESULT_FIELDS = [
    "log", "frac", "seed", "mode", "n_train_cases", "n_train_prefixes", "n_val_prefixes",
    "n_test_prefixes", "n_classes_train", "n_new_tokens", "test_unk_label_frac",
    "lr_enc", "lr_head", "batch_size", "tuned", "best_val_loss", "best_epoch",
    "epochs_run", "steps_run", "stopped_by_patience", "accuracy", "macro_f1", "train_time_s",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Modello e dati
# ----------------------------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, path, num_labels, sp_ids, vocab_size, pretrained, freeze):
        super().__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained(path, add_pooling_layer=False)
            self.bert.resize_token_embeddings(vocab_size)
        else:
            config = BertConfig.from_pretrained(path)  # stessa architettura
            config.vocab_size = vocab_size
            self.bert = BertModel(config, add_pooling_layer=False)  # pesi casuali

        self.cls_id = sp_ids["cls"]
        self.sep_id = sp_ids["sep"]
        self.pad_id = sp_ids["pad"]

        if freeze:
            for prm in self.bert.parameters():
                prm.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        mask = (
            (input_ids != self.cls_id)
            & (input_ids != self.sep_id)
            & (input_ids != self.pad_id)
        )
        mask = mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.classifier(pooled)


class PrefixStore:
    """Prefissi già tokenizzati in un unico array piatto int32 + offset (formato CSR)."""

    def __init__(self, flat, offsets):
        self.flat = flat
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i):
        return self.flat[self.offsets[i]:self.offsets[i + 1]]


class EncDataset(Dataset):
    def __init__(self, store, indices, labels):
        self.store = store
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.store.get(self.indices[i]), int(self.labels[i])


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        ids, labels = zip(*batch)
        length = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), length), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), length), dtype=torch.long)

        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
            attn[i, :len(x)] = 1

        return input_ids, attn, torch.tensor(labels, dtype=torch.long)


def encode_to_store(prefixes, tok2id, cls_id, sep_id, max_len):
    """[CLS] + ultime (max_len - 2) attività + [SEP], come il tokenizer con truncation_side="left".
    Equivalente al tokenizer perché ogni codice è UN token (garantito da build_token_map)."""
    body = max_len - 2
    n = len(prefixes)
    lens = np.fromiter((min(len(p), body) for p in prefixes), dtype=np.int64, count=n) + 2
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    flat = np.empty(int(offsets[-1]), dtype=np.int32)

    for i, p in enumerate(prefixes):
        tail = p[-body:] if len(p) > body else p
        o = int(offsets[i])
        flat[o] = cls_id
        flat[o + 1:o + 1 + len(tail)] = [tok2id[a] for a in tail]
        flat[o + 1 + len(tail)] = sep_id
    return PrefixStore(flat, offsets)


def build_token_map(tokenizer, codes):
    """Ogni codice attività e [UNK] devono diventare UN solo token. Restituisce (codice -> id, cls, sep)."""
    probe = list(codes) + [UNK]
    enc = tokenizer([[c] for c in probe], is_split_into_words=True)["input_ids"]
    unk_id = tokenizer.unk_token_id
    mapping, bad = {}, []

    for code, ids in zip(probe, enc):
        ok = len(ids) == 3 and (
            (ids[1] == unk_id) if code == UNK else (ids[1] != unk_id)
        )
        if ok:
            mapping[code] = ids[1]
        else:
            bad.append((code, ids))

    if bad:
        raise RuntimeError(
            f"Tokenizer: {len(bad)} token non gestiti correttamente, es. {bad[:3]}"
        )
    cls_id, sep_id = enc[0][0], enc[0][2]
    if cls_id != tokenizer.cls_token_id or sep_id != tokenizer.sep_token_id:
        raise RuntimeError("Tokenizer: id di [CLS]/[SEP] inattesi")
    return mapping, cls_id, sep_id


def build_optimizer(model, lr_enc, lr_head, wd):
    no_decay = ("bias", "LayerNorm.weight")
    groups = []

    for module, lr in ((model.bert, lr_enc), (model.classifier, lr_head)):
        named = [(name, prm) for name, prm in module.named_parameters() if prm.requires_grad]
        groups.append({
            "params": [prm for name, prm in named if not any(nd in name for nd in no_decay)],
            "weight_decay": wd,
            "lr": lr,
        })
        groups.append({
            "params": [prm for name, prm in named if any(nd in name for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        })

    return torch.optim.AdamW([group for group in groups if group["params"]])


def set_train(model, freeze):
    model.train()
    if freeze:
        model.bert.eval()  # encoder congelato: niente dropout


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    n_examples = 0

    for ids, attn, labels in loader:
        logits = model(ids.to(device), attn.to(device))
        total_loss += F.cross_entropy(
            logits, labels.to(device), reduction="sum"
        ).item()
        n_examples += len(labels)

    if n_examples == 0:
        raise RuntimeError("Il validation set è vuoto: impossibile calcolare la validation loss.")

    return total_loss / n_examples


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []

    for ids, attn, _ in loader:
        logits = model(ids.to(device), attn.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())

    return preds


# ----------------------------------------------------------------------------
# Training per epoche con early stopping
# ----------------------------------------------------------------------------
def train_one(mode_cfg, hp, ctx, seed, args):
    set_seed(seed)

    model = MeanPoolClassifier(
        args.pretrained,
        ctx["num_labels"],
        ctx["sp_ids"],
        ctx["vocab_size"],
        pretrained=mode_cfg["pretrained"],
        freeze=mode_cfg["freeze"],
    ).to(device)

    train_loader = DataLoader(
        ctx["train_ds"],
        batch_size=ctx["batch_size"],
        shuffle=True,
        collate_fn=ctx["collate"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        ctx["val_ds"],
        batch_size=args.eval_bs,
        shuffle=False,
        collate_fn=ctx["collate"],
        pin_memory=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Il training set è vuoto: impossibile avviare il training.")

    # Il DataLoader attraversa tutto il training set una volta per epoca.
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    opt = build_optimizer(model, hp["lr_enc"], hp["lr_head"], args.weight_decay)
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epochs_run = 0
    steps_run = 0
    best_state = None
    stopped_by_patience = False

    for epoch in range(1, args.epochs + 1):
        # ---- TRAIN: un passaggio completo sul training set ----
        set_train(model, mode_cfg["freeze"])
        epoch_loss_sum = 0.0
        epoch_examples = 0

        for ids, attn, labels in train_loader:
            batch_n = len(labels)
            micro = args.micro_bs if args.micro_bs > 0 else batch_n
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0

            for s in range(0, batch_n, micro):
                m_attn = attn[s:s + micro]
                width = int(m_attn.sum(1).max().item())  # toglie il padding in eccesso del micro-batch
                m_ids = ids[s:s + micro, :width].to(device)
                m_attn = m_attn[:, :width].to(device)
                m_labels = labels[s:s + micro].to(device)

                logits = model(m_ids, m_attn)
                # somma / batch_n: gradiente identico a quello della media sul batch intero
                loss = F.cross_entropy(logits, m_labels, reduction="sum") / batch_n
                loss.backward()
                batch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                [prm for prm in model.parameters() if prm.requires_grad], 1.0
            )
            opt.step()
            sched.step()

            epoch_loss_sum += batch_loss * batch_n
            epoch_examples += batch_n
            steps_run += 1

        epochs_run = epoch
        train_loss = epoch_loss_sum / max(epoch_examples, 1)

        # ---- VALIDATION: una valutazione alla fine di ogni epoca ----
        val_loss = evaluate_loss(model, val_loader)
        print(
            f"    Epoch {epoch}/{args.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| steps={steps_run}"
        )

        # ---- EARLY STOPPING: patience espressa in epoche ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                stopped_by_patience = True
                print(
                    f"    Early stopping all'epoca {epoch} "
                    f"({args.patience} epoche senza miglioramento)"
                )
                break

    if torch.cuda.is_available():
        print(f"    VRAM picco: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()

    del model, opt, sched, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "steps_run": steps_run,
        "stopped_by_patience": stopped_by_patience,
        "state": best_state,
        "hp": hp,
    }


# ----------------------------------------------------------------------------
# Utilità di orchestrazione
# ----------------------------------------------------------------------------
def list_pickles(pickle_dir, tune_seed, only_log):
    rx = re.compile(r"^(.*)_f([0-9.e+-]+)_s(\d+)\.pkl$")
    items = []

    for filename in os.listdir(pickle_dir):
        match = rx.match(filename)
        if not match or (only_log and only_log not in match.group(1)):
            continue

        log_name = match.group(1)
        frac = float(match.group(2))
        seed = int(match.group(3))
        items.append(((log_name, frac, seed != tune_seed, seed), os.path.join(pickle_dir, filename)))

    return [path for _, path in sorted(items)]


def load_done(csv_path):
    if not os.path.exists(csv_path):
        return set()

    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        required = {"log", "frac", "seed", "mode", "best_epoch", "epochs_run"}

        if not required.issubset(fields):
            raise RuntimeError(
                f"Il file {csv_path} sembra appartenere al vecchio protocollo a step. "
                "Per evitare di mescolare risultati non confrontabili, usa un nuovo "
                "--results_csv (ad esempio risultati_nap_fractions_epoche.csv)."
            )

        return {
            (row["log"], round(float(row["frac"]), 6), int(row["seed"]), row["mode"])
            for row in reader
        }


def append_row(csv_path, row):
    new_file = not os.path.exists(csv_path)

    if not new_file:
        with open(csv_path, "r", newline="") as fh:
            existing_fields = next(csv.reader(fh), [])
        if existing_fields != RESULT_FIELDS:
            raise RuntimeError(
                f"Intestazione di {csv_path} non compatibile con questo script. "
                "Usa un nuovo file --results_csv per i risultati in epoche."
            )

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in RESULT_FIELDS})


def main():
    ap = argparse.ArgumentParser(
        description="Confronto scratch/pretrained con training espresso in epoche."
    )
    ap.add_argument(
        "--pickle_dir",
        default="/home/vincenzo/Documenti/processbert/prepare_data/log_nap_frac/pkl",
    )
    ap.add_argument("--tokenizer_dir", default="tokenizer")
    ap.add_argument(
        "--pretrained",
        default="best_epoch_004",
        help="checkpoint SENSEI (anche per la config)",
    )
    ap.add_argument("--out_dir", default="./nap_fraction_runs")
    ap.add_argument(
        "--results_csv",
        default="risultati_nap_fractions_epoche.csv",
    )
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--only_log", default=None)
    ap.add_argument("--tune_seed", type=int, default=0)
    ap.add_argument(
        "--no_tune",
        action="store_true",
        help="usa solo la configurazione centrale della griglia",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="numero massimo di epoche di training",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=3,
        help="numero di epoche consecutive senza miglioramento",
    )
    ap.add_argument("--max_bs", type=int, default=256)
    ap.add_argument("--min_bs", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--micro_bs", type=int, default=0,
                    help="se > 0, spezza ogni batch in micro-batch di questa dimensione (meno VRAM, stesso gradiente)")
    ap.add_argument("--eval_bs", type=int, default=256, help="batch di validation/test (non cambia i risultati)")
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    if args.epochs < 1:
        ap.error("--epochs deve essere almeno 1")
    if args.patience < 1:
        ap.error("--patience deve essere almeno 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        ap.error("--warmup_ratio deve essere compreso tra 0 e 1 (1 escluso)")
    if args.min_bs < 1 or args.max_bs < args.min_bs:
        ap.error("Specificare batch size validi: 1 <= min_bs <= max_bs")

    os.makedirs(args.out_dir, exist_ok=True)

    # Cache separata: evita di riutilizzare lr scelti con il precedente protocollo a step.
    cfg_path = os.path.join(args.out_dir, "best_cfg_epochs.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as fh:
            cfg_cache = json.load(fh)
    else:
        cfg_cache = {}

    done = load_done(args.results_csv)
    pretrained_vocab = BertConfig.from_pretrained(args.pretrained).vocab_size
    print(f"Device: {device} | pickle: {args.pickle_dir}")
    print(
        f"Protocollo: max {args.epochs} epoche | patience={args.patience} epoche | "
        "validation a fine epoca"
    )

    for pkl_path in list_pickles(args.pickle_dir, args.tune_seed, args.only_log):
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        for unused in ("X_train_txt", "X_test_txt", "len_train", "len_test", "groups_test"):
            data.pop(unused, None)
        gc.collect()

        meta = data["meta"]
        log = meta["log"]
        frac = round(float(meta["frac"]), 6)
        seed = int(meta["seed"])

        todo = [mode for mode in args.modes if (log, frac, seed, mode) not in done]
        if not todo:
            continue

        print(
            "\n" + "=" * 80
            + f"\n{log} | frac={frac} | seed={seed} | modes={todo}\n"
            + "=" * 80
        )

        # --- tokenizer (mapping specifico del sottoinsieme) ---
        tokenizer = BertTokenizerFast.from_pretrained(
            args.tokenizer_dir, truncation_side="left"
        )
        codes = list(data["activity_map"].values())
        tokenizer.add_tokens(codes)
        tok2id, cls_id, sep_id = build_token_map(tokenizer, codes)
        n_new_tokens = len(tokenizer) - pretrained_vocab
        sp_ids = {
            "cls": tokenizer.cls_token_id,
            "sep": tokenizer.sep_token_id,
            "pad": tokenizer.pad_token_id,
        }

        # --- label: solo classi viste nel train ---
        classes = sorted(set(data["y_train_abs"]))
        label2id = {label: i for i, label in enumerate(classes)}
        id2label = {i: label for label, i in label2id.items()}
        y_all = [label2id[label] for label in data["y_train_abs"]]

        # --- split train/val per CASE ---
        groups = np.asarray(data["groups_train"], dtype=object)
        if len(set(groups.tolist())) < 2:
            raise RuntimeError("Servono almeno 2 case nel train per lo split train/val")

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=args.val_ratio,
            random_state=seed,
        )
        tr_idx, va_idx = next(
            gss.split(np.arange(len(y_all)), groups=groups)
        )

        # prefissi -> array piatto int32 (lookup diretta, senza tokenizer); poi si liberano le liste grezze
        X_all = encode_to_store(data["X_train_abs"], tok2id, cls_id, sep_id, args.max_len)
        X_test = encode_to_store(data["X_test_abs"], tok2id, cls_id, sep_id, args.max_len)
        data.pop("X_train_abs")
        data.pop("X_test_abs")
        gc.collect()
        print(f"  RAM picco del processo: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB")

        y_all = np.asarray(y_all, dtype=np.int64)
        train_ds = EncDataset(X_all, tr_idx, y_all[tr_idx])
        val_ds = EncDataset(X_all, va_idx, y_all[va_idx])
        test_ds = EncDataset(X_test, np.arange(len(X_test)), np.zeros(len(X_test), dtype=np.int64))

        collate = Collator(sp_ids["pad"])
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_bs,
            shuffle=False,
            collate_fn=collate,
            pin_memory=True,
        )

        batch_size = int(
            np.clip(len(train_ds) // 20, args.min_bs, args.max_bs)
        )
        ctx = {
            "num_labels": len(classes),
            "sp_ids": sp_ids,
            "vocab_size": len(tokenizer),
            "train_ds": train_ds,
            "val_ds": val_ds,
            "collate": collate,
            "batch_size": batch_size,
        }

        test_unk = float(np.mean([label == UNK for label in data["y_test_abs"]]))
        print(
            f"  classi train={len(classes)} | train={len(train_ds)} "
            f"val={len(va_idx)} test={len(X_test)} | bs={batch_size} "
            f"| token nuovi={n_new_tokens} "
            f"| label test sconosciute={test_unk:.1%} "
            f"| batch/epoca={int(np.ceil(len(train_ds) / batch_size))}"
        )

        y_true = data["y_test_txt"]
        eval_labels = sorted(set(y_true))
        inv = data["inverse_map"]

        for mode in todo:
            mode_cfg = MODES[mode]
            key = f"{log}|{frac:g}|{mode}"
            grid = mode_cfg["grid"]
            tuned = False

            if key in cfg_cache:
                grid = [cfg_cache[key]]
            elif args.no_tune:
                grid = [grid[len(grid) // 2]]
            else:
                tuned = True

            t0 = time.time()
            best_run = None

            for hp in grid:
                run = train_one(mode_cfg, hp, ctx, seed, args)
                print(
                    f"  [{mode}] lr_enc={hp['lr_enc']:g} "
                    f"lr_head={hp['lr_head']:g} "
                    f"-> val_loss={run['best_val_loss']:.4f} "
                    f"@epoch {run['best_epoch']} "
                    f"(run {run['epochs_run']} epoche, "
                    f"{run['steps_run']} step, "
                    f"patience={run['stopped_by_patience']})"
                )

                if best_run is None or run["best_val_loss"] < best_run["best_val_loss"]:
                    best_run = run

            if best_run is None or best_run["state"] is None:
                raise RuntimeError(f"Nessun checkpoint valido prodotto per {log}, {mode}")

            if tuned:
                cfg_cache[key] = best_run["hp"]
                with open(cfg_path, "w") as fh:
                    json.dump(cfg_cache, fh, indent=1)

            # --- test con il best checkpoint, selezionato SOLO sulla validation ---
            model = MeanPoolClassifier(
                args.pretrained,
                ctx["num_labels"],
                sp_ids,
                ctx["vocab_size"],
                pretrained=mode_cfg["pretrained"],
                freeze=mode_cfg["freeze"],
            ).to(device)
            model.load_state_dict(best_run["state"])

            preds = predict(model, test_loader)
            pred_names = [inv.get(id2label[i], UNK) for i in preds]
            acc = accuracy_score(y_true, pred_names)
            f1 = f1_score(
                y_true,
                pred_names,
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
            print(f"  [{mode}] TEST acc={acc:.4f} macro-F1={f1:.4f}")

            if args.save_models:
                model_path = os.path.join(
                    args.out_dir,
                    f"{log}_f{frac:g}_s{seed}_{mode}_epochs.pt",
                )
                torch.save(best_run["state"], model_path)

            append_row(
                args.results_csv,
                {
                    "log": log,
                    "frac": frac,
                    "seed": seed,
                    "mode": mode,
                    "n_train_cases": meta["n_train_cases"],
                    "n_train_prefixes": len(train_ds),
                    "n_val_prefixes": len(va_idx),
                    "n_test_prefixes": len(X_test),
                    "n_classes_train": len(classes),
                    "n_new_tokens": n_new_tokens,
                    "test_unk_label_frac": test_unk,
                    "lr_enc": best_run["hp"]["lr_enc"],
                    "lr_head": best_run["hp"]["lr_head"],
                    "batch_size": batch_size,
                    "tuned": tuned,
                    "best_val_loss": best_run["best_val_loss"],
                    "best_epoch": best_run["best_epoch"],
                    "epochs_run": best_run["epochs_run"],
                    "steps_run": best_run["steps_run"],
                    "stopped_by_patience": best_run["stopped_by_patience"],
                    "accuracy": acc,
                    "macro_f1": f1,
                    "train_time_s": round(time.time() - t0, 1),
                },
            )

            del model, best_run
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del data, X_all, X_test, train_ds, val_ds, test_ds, test_loader, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nCompletato. Risultati in", args.results_csv)


if __name__ == "__main__":
    main()
