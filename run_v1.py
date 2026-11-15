"""New train/evaluate wrapper; not the recovered original experiment scheduler."""
import argparse
import csv
import json
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from config import model_args, MAX_EPOCHS, MIN_EPOCHS, PATIENCE
from config import FUSION_LR, BACKBONE_LR, WEIGHT_DECAY
from data import load_manifest, load_pair


def metrics(rows):
    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob_positive"] for r in rows])
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                acc=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bag(row, device):
    x, c = load_pair(row["high_pkl"], row["low_pkl"])
    return torch.from_numpy(x).to(device), torch.from_numpy(c).to(device)


@torch.no_grad()
def evaluate(model, manifest, device, high_only=False):
    model.eval()
    predictions = []
    for row in manifest:
        x, c = bag(row, device)
        if high_only:
            _, prob = model._high_fwd(x[:, :model.input_dim])
        else:
            _, prob, _, _, _ = model(x, coords=c)
        predictions.append(dict(slide_id=row["slide_id"], label=row["label"],
                                prob_positive=float(prob[0, 1].item())))
    return predictions, metrics(predictions)


def save_results(folder, name, rows, scores):
    with (folder / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "label", "prob_positive"])
        writer.writeheader()
        writer.writerows(rows)
    (folder / f"{name}_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")


def assert_separate(*splits):
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            for key in ("slide_id", "patient_id"):
                av = {r.get(key) for r in a} - {None, ""}
                bv = {r.get(key) for r in b} - {None, ""}
                if av & bv:
                    raise ValueError(f"Overlapping {key} across splits: {sorted(av & bv)[:3]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--high-ckpt", type=Path, required=True)
    parser.add_argument("--v1-ckpt", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train" and not (args.train and args.val):
        parser.error("train mode requires --train and --val")
    if args.mode == "evaluate" and not (args.v1_ckpt and args.test):
        parser.error("evaluate mode requires --v1-ckpt and --test")
    if args.mode == "train" and args.v1_ckpt:
        parser.error("Training starts from High; --v1-ckpt is for evaluation only")
    for rel in ["model_vit_pmt_clu.py", "nystrom_attention_custom.py",
                "tcformer/tcformer_layers.py", "tcformer/tcformer_utils.py",
                "tcformer/transformer_utils.py"]:
        if not (Path(__file__).parent / "models" / rel).is_file():
            parser.error(f"Missing original backbone dependency: models/{rel}; see README.md")
    from models.model_bialign_ms import BiAlignMS_PTCMIL

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Each run has its own directory; do not overwrite previous predictions.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "run_config.json").write_text(json.dumps({**vars(args), "model": vars(model_args())},
                                            default=str, indent=2), encoding="utf-8")
    device = torch.device(args.device)
    model = BiAlignMS_PTCMIL(str(args.high_ckpt), args=model_args()).to(device)
    test = load_manifest(args.test) if args.test else []
    if args.mode == "evaluate":
        payload = torch.load(args.v1_ckpt, map_location="cpu")
        state = payload
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in payload:
                state = payload[key]
                break
        model.load_state_dict(state, strict=True)
    else:
        train, val = load_manifest(args.train), load_manifest(args.val)
        assert_separate(train, val, test)
        if {r['label'] for r in val} != {0, 1}:
            raise ValueError("Validation AUC selection requires both classes")
        _, reference = evaluate(model, val, device, high_only=True)
        fast, slow = model.get_trainable_params()
        optimizer = torch.optim.Adam([
            {"params": [p for _, p in fast], "lr": FUSION_LR},
            {"params": [p for _, p in slow], "lr": BACKBONE_LR},
        ], weight_decay=WEIGHT_DECAY)
        def snapshot():
            return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_state = snapshot()
        best_epoch, best_auc, stale = -1, -1.0, 0
        rng = np.random.default_rng(args.seed)
        history = []
        for epoch in range(MAX_EPOCHS):
            model.train()
            for index in rng.permutation(len(train)):
                row = train[index]
                x, c = bag(row, device)
                y = torch.tensor([row["label"]], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits, _, _, _, _ = model(x, coords=c)
                torch.nn.functional.cross_entropy(logits, y).backward()
                optimizer.step()
            _, score = evaluate(model, val, device)
            qualified = score["acc"] + 1e-12 >= reference["acc"] and score["fp"] <= reference["fp"]
            improved = qualified and score["auc"] > best_auc + 1e-12
            if improved:
                best_state, best_epoch, best_auc, stale = snapshot(), epoch, score["auc"], 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, qualified=qualified, improved=improved, **score))
            print(json.dumps(history[-1]), flush=True)
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break
        model.load_state_dict(best_state, strict=True)
        torch.save(dict(model_state=best_state, best_epoch=best_epoch,
                        fallback_to_initial=(best_epoch == -1)), args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.out / "selection.json").write_text(json.dumps(dict(
            high_val=reference, best_epoch=best_epoch, fallback_to_initial=(best_epoch == -1)), indent=2), encoding="utf-8")
        rows, scores = evaluate(model, val, device)
        save_results(args.out, "val", rows, scores)
    if test:
        rows, scores = evaluate(model, test, device)
        save_results(args.out, "test", rows, scores)


if __name__ == "__main__":
    main()
