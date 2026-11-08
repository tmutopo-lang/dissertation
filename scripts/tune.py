"""
tune.py
───────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Design (Chapter 4: hyperparameter search)

Bayesian hyperparameter search (Optuna, Tree-structured Parzen Estimator) for
the chain-sequence models. Every architecture gets the same number of trials,
so no model is favoured by a larger tuning budget.

Each trial trains on the training partition and is scored on the VALIDATION
partition only (the test partition is never touched here):

  objective = validation recall at the threshold that keeps validation FPR
              <= --target-fpr, plus 0.01 x validation AUC as a tie-breaker

The best configuration per architecture is written to
results/tuning/<arch>.json together with the train_model.py command that
reproduces it for the final multi-seed runs.

Search space:
  units per direction  {64, 128, 256}
  recurrent layers     {1, 2}
  attention heads      {4, 8}
  dropout              0.1 - 0.5
  learning rate        1e-4 - 3e-3 (log scale)
  batch size           {64, 128}
  head widths          {(256, 128), (128, 64)}

Usage:
  pip install optuna
  python scripts/tune.py --arch hybrid --trials 20
  python scripts/tune.py --arch all --trials 20 --epochs 20
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import optuna
import keras

from eval_common import calibrate, metrics, temporal_split
from train_model import build, load, standardise

ARCHS = ["hybrid", "bilstm", "lstm", "lstm_attn", "attn", "cnn"]
HEADS = {"256-128": (256, 128), "128-64": (128, 64)}


def make_objective(arch, Xs, mask, y, tr, va, args):
    T, F = Xs.shape[1:]
    pos = max(y[tr].mean(), 1e-6)
    cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    def objective(trial):
        p = {
            "units": trial.suggest_categorical("units", [64, 128, 256]),
            "layers": trial.suggest_categorical("layers", [1, 2]),
            "heads": trial.suggest_categorical("heads", [4, 8]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch": trial.suggest_categorical("batch", [64, 128]),
            "dense": trial.suggest_categorical("dense", list(HEADS)),
        }
        keras.utils.set_random_seed(args.seed)
        model, _ = build(arch, T, F, p["units"], p["layers"], p["heads"], p["dropout"],
                         p["lr"], HEADS[p["dense"]])
        cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=args.patience, restore_best_weights=True)]
        model.fit([Xs[tr], mask[tr]], y[tr], validation_data=([Xs[va], mask[va]], y[va]),
                  epochs=args.epochs, batch_size=p["batch"], class_weight=cw,
                  callbacks=cb, verbose=0)
        s_va = model.predict([Xs[va], mask[va]], batch_size=512, verbose=0).ravel()
        thr = calibrate(y[va], s_va, args.target_fpr)
        m = metrics(y[va], s_va, thr)
        trial.set_user_attr("val_fpr", m["fpr"])
        trial.set_user_attr("val_recall", m["recall"])
        trial.set_user_attr("val_auc", m["auc"])
        keras.backend.clear_session()
        return m["recall"] + 0.01 * m["auc"]

    return objective


def main():
    ap = argparse.ArgumentParser(description="Bayesian hyperparameter search (validation only)")
    ap.add_argument("--data", default="data/dataset/dataset.npz")
    ap.add_argument("--arch", default="hybrid", choices=ARCHS + ["all"])
    ap.add_argument("--trials", type=int, default=20, help="same budget for every architecture")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tuning")
    args = ap.parse_args()

    d = load(args.data)
    X, L, y, src = d["X"], d["lengths"].astype(int), d["y"].astype(int), d["source"]
    tr, va, _ = temporal_split(len(y), src)
    Xs, mask, _, _ = standardise(X, L, tr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for arch in (ARCHS if args.arch == "all" else [args.arch]):
        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(make_objective(arch, Xs, mask, y, tr, va, args), n_trials=args.trials,
                       callbacks=[lambda s, t: print(
                           f"  [{arch}] trial {t.number + 1}/{args.trials}  "
                           f"recall {t.user_attrs.get('val_recall', 0):.3f}  "
                           f"fpr {t.user_attrs.get('val_fpr', 0):.3f}  {t.params}")])
        b = study.best_trial
        cmd = (f"python scripts/train_model.py --arch {arch} --units {b.params['units']} "
               f"--layers {b.params['layers']} --heads {b.params['heads']} "
               f"--dropout {b.params['dropout']:.3f} --lr {b.params['lr']:.5f} "
               f"--batch {b.params['batch']} --dense {' '.join(map(str, HEADS[b.params['dense']]))}")
        res = {"arch": arch, "trials": args.trials, "best_params": b.params,
               "best_validation": b.user_attrs, "command": cmd,
               "minutes": round((time.time() - t0) / 60, 1),
               "all_trials": [{"params": t.params, **t.user_attrs, "value": t.value}
                              for t in study.trials]}
        (out / f"{arch}.json").write_text(json.dumps(res, indent=2))
        print(f"\n[✓] {arch}: best validation recall {b.user_attrs['val_recall']:.3f} "
              f"at FPR {b.user_attrs['val_fpr']:.3f}  ({res['minutes']} min)")
        print(f"    {cmd}\n")


if __name__ == "__main__":
    main()
