import argparse
import os
import pickle

import numpy as np
import pandas as pd

#CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"
CASE, ACT, TS = "Case ID", "Activity", "time:timestamp"

UNK = "[UNK]"

import pm4py  # pip install pm4py
# ----------------------------------------------------------------------------
# Lettura e split
# ----------------------------------------------------------------------------
def load_log(path):
    df = pm4py.read_xes(path) if path.endswith(".xes") else pd.read_csv(path, sep=";")
    #df = pd.read_csv(path, sep=",")
    df[ACT] = df[ACT].astype(str)
    df[TS] = pd.to_datetime(df[TS], utc=True)  # ordinamento temporale vero, non alfabetico
    return df.sort_values(TS, kind="mergesort").reset_index(drop=True)


def chronological_split(df, train_ratio):
    """Split per case in base al timestamp del primo evento (come nello script originale)."""
    start = df.groupby(CASE)[TS].min().reset_index().sort_values(TS, kind="mergesort")
    ids = start[CASE].tolist()
    n_train = int(train_ratio * len(ids))
    train_ids, test_ids = ids[:n_train], ids[n_train:]
    assert not set(train_ids) & set(test_ids)
    train = df[df[CASE].isin(set(train_ids))]
    test = df[df[CASE].isin(set(test_ids))]
    return train, test, train_ids


# ----------------------------------------------------------------------------
# Sottocampionamento, mapping, prefissi
# ----------------------------------------------------------------------------
def n_cases_for(frac, n_total, min_cases):
    if frac >= 1.0:
        return n_total
    return min(n_total, max(int(round(frac * n_total)), min_cases))


def build_mapping(train_df, code_fmt):
    """Rank di frequenza (eventi) sul train sottocampionato. Pareggi: ordine alfabetico.
    code_fmt deve coincidere con il formato usato nel pretraining, es. "ACT_{:03d}"."""
    counts = train_df[ACT].value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], a))
    return {a: code_fmt.format(i + 1) for i, a in enumerate(order)}


def to_traces(df):
    """case id -> lista di attività ordinate nel tempo (df è già ordinato per timestamp)."""
    return df.groupby(CASE, sort=False)[ACT].apply(list)


def make_prefixes(traces):
    X, y, case_ids, lens = [], [], [], []
    for cid, acts in traces.items():
        for j in range(1, len(acts)):
            X.append(acts[:j])
            y.append(acts[j])
            case_ids.append(cid)
            lens.append(j)
    return X, y, case_ids, lens


def apply_mapping(X, y, amap):
    X_abs = [[amap.get(a, UNK) for a in prefix] for prefix in X]
    y_abs = [amap.get(a, UNK) for a in y]
    return X_abs, y_abs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_name")
    ap.add_argument("--logs_dir", default="/home/vincenzo/Documenti/processbert/prepare_data/logs")
    ap.add_argument("--out_dir", default="/home/vincenzo/Documenti/processbert/prepare_data/log_nap_frac/pkl")
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--min_cases", type=int, default=10)
    ap.add_argument("--train_ratio", type=float, default=0.66)
    ap.add_argument("--code_fmt", default="ACT_{:03d}",
                    help='formato del codice attività, deve combaciare col pretraining (es. "ACT_{:03d}" o "Activity{}")')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.dataset_name
    print(f"== {name}")

    df = load_log(os.path.join(args.logs_dir, f"{name}.csv"))
    train_df, test_df, train_ids_ordered = chronological_split(df, args.train_ratio)
    n_total = len(train_ids_ordered)
    print(f"Case train: {n_total} | case test: {test_df[CASE].nunique()}")

    # Test set: calcolato UNA volta, identico per tutte le configurazioni
    X_test_txt, y_test_txt, groups_test, len_test = make_prefixes(to_traces(test_df))
    label_set_txt = sorted(df[ACT].unique())  # set di classi fisso (log intero) per le metriche

    for seed in args.seeds:
        # Una permutazione per seed: prendendo i primi n si ottengono sottoinsiemi annidati
        perm = np.random.RandomState(seed).permutation(np.array(train_ids_ordered, dtype=object))
        seen_n = set()

        for frac in args.fracs:
            n = n_cases_for(frac, n_total, args.min_cases)

            if n in seen_n:
                print(f"  [skip] frac={frac} seed={seed}: n={n} già generato (min_cases)")
                continue
            seen_n.add(n)
            if n == n_total and seed != args.seeds[0]:
                continue  # il train completo è deterministico: un solo pickle

            keep = set(perm[:n].tolist())
            sub_train = train_df[train_df[CASE].isin(keep)]  # mantiene l'ordine temporale

            # Mapping SOLO sul train sottocampionato
            amap = build_mapping(sub_train, args.code_fmt)
            inverse_map = {v: k for k, v in amap.items()}

            X_train_txt, y_train_txt, groups_train, len_train = make_prefixes(to_traces(sub_train))
            X_train_abs, y_train_abs = apply_mapping(X_train_txt, y_train_txt, amap)
            X_test_abs, y_test_abs = apply_mapping(X_test_txt, y_test_txt, amap)

            n_unk_test = sum(a == UNK for a in y_test_abs)
            print(
                f"  frac={frac:<5} seed={seed} | case={n:<6} prefissi train={len(X_train_abs):<8} "
                f"attività train={len(amap):<4} | test: {len(X_test_abs)} prefissi, "
                f"{n_unk_test} label sconosciute ({n_unk_test / max(len(y_test_abs), 1):.1%})"
            )

            out = {
                "X_train_abs": X_train_abs, "y_train_abs": y_train_abs,
                "X_train_txt": X_train_txt, "y_train_txt": y_train_txt,
                "X_test_abs": X_test_abs, "y_test_abs": y_test_abs,
                "X_test_txt": X_test_txt, "y_test_txt": y_test_txt,
                "activity_map": amap,          # attività -> codice (es. ACT_001) (solo train sottocampionato)
                "inverse_map": inverse_map,    # codice -> attività originale
                "label_set_txt": label_set_txt,
                "len_train": len_train, "len_test": len_test,
                "groups_train": groups_train,  # case id di ogni prefisso (per GroupShuffleSplit)
                "groups_test": groups_test,
                "meta": {"log": name, "frac": frac, "seed": seed,
                         "n_train_cases": n, "n_total_train_cases": n_total,
                         "train_ratio": args.train_ratio, "min_cases": args.min_cases},
            }
            out_path = os.path.join(args.out_dir, f"{name}_f{frac:g}_s{seed}.pkl")
            with open(out_path, "wb") as fh:
                pickle.dump(out, fh)


if __name__ == "__main__":
    main()
