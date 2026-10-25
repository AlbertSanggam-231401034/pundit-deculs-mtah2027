"""Latihan eksternal IDSSE: J03WMX -> validasi J03WN1.

Tidak membaca aset atau GT challenge. Tidak memakai model eksperimen terdahulu.
Konfigurasi awal tetap, tanpa pencarian parameter. Output bukan submission.
Data: DFL; Bassek et al. (2025), doi:10.6084/m9.figshare.28196177.v1,
CC BY 4.0. Memerlukan numpy, pandas, scikit-learn, joblib, threadpoolctl.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm

NAMES = (
    "ball_observed", "ball_displacement_speed_1s_mps", "nearest_home_m",
    "nearest_away_m", "nearest_any_m", "away_minus_home_distance_m",
    "abs_distance_margin_m", "home_visible", "away_visible",
    "players_displacement_speed_1s_median_mps",
    "players_displacement_speed_1s_p75_mps", "players_motion_samples",
    "home_spread_median_m", "away_spread_median_m", "centroid_separation_m",
)
CONFIG = dict(learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
              min_samples_leaf=100, l2_regularization=1.0,
              class_weight="balanced", early_stopping=False, random_state=2027)
TEAM_RADIUS = 50  # Official reference's fixed 2-second radius, not public-GT selection.
STATE_RADIUS = 12  # 25-frame centred probability average (~1 second).
FPS = 25


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(folder, expected_match):
    folder = Path(folder)
    with np.load(folder / "features.npz", allow_pickle=False) as f:
        X = f["X"].copy()
        names = tuple(f["feature_names"].tolist())
        frame, half = f["frame"].copy(), f["half"].copy()
        match = str(f["match_id"])
    with np.load(folder / "targets.npz", allow_pickle=False) as t:
        if str(t["match_id"]) != match or not np.array_equal(t["frame"], frame) or not np.array_equal(t["half"], half):
            raise ValueError("Fitur dan target tidak sejajar.")
        y = t["y"].copy()
    if match != expected_match or names != NAMES or X.shape != (len(y), len(NAMES)):
        raise ValueError("Match atau schema fitur tidak sesuai.")
    if np.isinf(X).any() or set(y.tolist()) - set("HADX"):
        raise ValueError("Nilai fitur/label tidak valid.")
    if set(half.tolist()) != {1, 2}:
        raise ValueError("Babak tidak lengkap.")
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)) or not np.all(np.diff(frame[idx]) == 1):
            raise ValueError("Frame/babak terputus atau tidak berurutan.")
    hashes = {name: sha256(folder / name) for name in ("features.npz", "targets.npz")}
    return X, y, frame, half, hashes


def mirror_sides(X):
    """Remove systematic home/away bias from LIVE/D training by swapping sides."""
    out = X.copy()
    for a, b in ((2, 3), (7, 8), (12, 13)):
        out[:, [a, b]] = X[:, [b, a]]
    out[:, 5] = -X[:, 5]
    return out


def centred_mean(values, radius):
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    idx = np.arange(len(values))
    left = np.maximum(0, idx - radius)
    right = np.minimum(len(values), idx + radius + 1)
    return (prefix[right] - prefix[left]) / (right - left)


def nearest_reference(X, half):
    h, a = X[:, 2], X[:, 3]
    available = np.isfinite(h) | np.isfinite(a)
    codes = np.full(len(X), -1, dtype=np.int8)
    codes[available] = np.where(np.where(np.isfinite(h[available]), h[available], np.inf)
                               <= np.where(np.isfinite(a[available]), a[available], np.inf), 0, 1)
    labels = np.full(len(X), "D", dtype="<U1")
    for period in (1, 2):
        idx = np.flatnonzero(half == period)
        votes_h = centred_mean((codes[idx] == 0).astype(float), TEAM_RADIUS)
        votes_a = centred_mean((codes[idx] == 1).astype(float), TEAM_RADIUS)
        seen = (votes_h + votes_a) > 0
        labels[idx[seen]] = np.where(votes_h[seen] >= votes_a[seen], "H", "A")
    return labels


def carry_team(reference, half):
    """Fill missing team evidence within a half, never with ground-truth ownership."""
    out = reference.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        window = reference[idx]
        known = np.isin(window, ["H", "A"])
        if not known.any():
            out[idx] = "X"
            continue
        carry = np.maximum.accumulate(np.where(known, np.arange(len(idx)), -1))
        carry[carry < 0] = np.flatnonzero(known)[0]
        out[idx] = window[carry]
    return out


def predict_states(model, X, half):
    positive = list(model.classes_).index(1)
    p = model.predict_proba(X)[:, positive]
    dead = np.zeros(len(X), dtype=bool)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        dead[idx] = centred_mean(p[idx], STATE_RADIUS) >= 0.5
    return dead


def predictions(models, X, half):
    ref = nearest_reference(X, half)
    team = carry_team(ref, half)
    full_dead = predict_states(models["full"], X, half)
    player_dead = predict_states(models["players"], X[:, 7:], half)
    missing = X[:, 0] == 0
    routed_dead = np.where(missing, player_dead, full_dead)
    return {
        "nearest_reference2s": ref,
        "external_full": np.where(routed_dead, "D", team),
        "external_players_only": np.where(player_dead, "D", team),
    }


def hide_ball(X, half):
    """Fixed, label-blind 5s gaps in every 20s; preserve player features."""
    masked = X.copy()
    hidden = np.zeros(len(X), dtype=bool)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        relative = np.arange(len(idx))
        off = (relative % (20 * FPS) >= 10 * FPS) & (relative % (20 * FPS) < 15 * FPS)
        hidden[idx] = off
        # The 1-second displacement also becomes unavailable when its past endpoint was hidden.
        bad_speed = off.copy()
        bad_speed[FPS:] |= off[:-FPS]
        masked[idx[bad_speed], 1] = np.nan
    masked[hidden, 0] = 0
    masked[hidden, 1:7] = np.nan
    return masked, hidden


def padded_codes(labels, half):
    # A one-frame X gap prevents the scorer from joining possessions across halftime.
    parts = [pm.label_codes(labels[half == 1]), np.array([pm.X], dtype=np.int8),
             pm.label_codes(labels[half == 2])]
    return np.concatenate(parts)


def summary(y, pred, half):
    r = pm.evaluate(padded_codes(y, half), padded_codes(pred, half))
    return {"macroF1": r["macro_f1"], "spellF1": r["spell_f1_mean"],
            "F1_D": r["per_class"]["D"]["f1"],
            "precision_D": r["per_class"]["D"]["precision"],
            "recall_D": r["per_class"]["D"]["recall"],
            "possessions": r["n_pred_spells"], "gt_possessions": r["n_gt_spells"],
            "false_D": int(np.sum(np.isin(y, ["H", "A"]) & (pred == "D"))),
            "missed_D": int(np.sum((y == "D") & (pred != "D")))}


def missing_summary(y, pred, hidden):
    valid = hidden & np.isin(y, ["H", "A", "D"])
    dead, called = y[valid] == "D", pred[valid] == "D"
    tp, fp, fn = int(np.sum(dead & called)), int(np.sum(~dead & called)), int(np.sum(dead & ~called))
    live = ~dead
    return {"frames": int(valid.sum()), "F1_D": 2 * tp / max(1, 2 * tp + fp + fn),
            "recall_LIVE": int(np.sum(live & np.isin(pred[valid], ["H", "A"]))) / max(1, int(live.sum())),
            "LIVE_frames": int(live.sum()), "D_frames": int(dead.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/external/idsse"
    parser.add_argument("--train-dir", type=Path, default=base / "J03WMX/features_v1")
    parser.add_argument("--valid-dir", type=Path, default=base / "J03WN1/features_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation/experiments/idsse_external_v1")
    args = parser.parse_args()
    X, y, _, half_train, hashes_train = load_data(args.train_dir, "DFL-MAT-J03WMX")
    known = np.isin(y, ["H", "A", "D"])
    Xtrain = np.concatenate([X[known], mirror_sides(X[known])])
    binary = np.tile((y[known] == "D").astype(np.int8), 2)
    print("LATIH: J03WMX (Koln-Bayern)")
    print("Frame asli:", int(known.sum()), "| target:", dict(Counter(y[known].tolist())))
    print("Augmentasi tukar sisi hanya pada data latih. Konfigurasi tetap; tanpa tuning.", flush=True)
    models = {}
    with threadpool_limits(limits=4):
        for name, inputs in (("full", Xtrain), ("players", Xtrain[:, 7:])):
            print("Melatih model LIVE/D:", name, flush=True)
            models[name] = HistGradientBoostingClassifier(**CONFIG).fit(inputs, binary)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "models.joblib"
    joblib.dump({"models": models, "config": CONFIG, "feature_names": NAMES,
                 "train_match": "DFL-MAT-J03WMX", "team_radius": TEAM_RADIUS,
                 "state_radius": STATE_RADIUS}, model_path)
    # Validation is loaded only AFTER both models have been fitted and saved.
    V, truth, frame, half, hashes_valid = load_data(args.valid_dir, "DFL-MAT-J03WN1")
    print("\nVALIDASI: J03WN1 (Bochum-Leverkusen),", len(V), "frame", flush=True)
    rows, pred_store = [], {}
    masked, hidden = hide_ball(V, half)
    with threadpool_limits(limits=4):
        for scenario, data in (("bola_tersedia", V), ("bola_disembunyikan_5s", masked)):
            preds = predictions(models, data, half)
            print("\n" + scenario)
            print("method                    macroF1 spellF1   F1_D  poss false_D missed_D")
            for name, pred in preds.items():
                stats = summary(truth, pred, half)
                row = {"scenario": scenario, "method": name, **stats}
                if scenario != "bola_tersedia":
                    row["hidden_frames_only"] = missing_summary(truth, pred, hidden)
                rows.append(row)
                pred_store[scenario + "__" + name] = pred
                print("{:<25} {:.4f}  {:.4f} {:.4f} {:>5} {:>7} {:>8}".format(
                    name, stats["macroF1"], stats["spellF1"], stats["F1_D"],
                    stats["possessions"], stats["false_D"], stats["missed_D"]))
    print("\nJUMLAH POSSESSION ACUAN EKSTERNAL:", rows[0]["gt_possessions"])
    print("UJI HANYA FRAME BOLA DISEMBUNYIKAN (sintetis):", int(hidden.sum()), "frame")
    for row in rows:
        if "hidden_frames_only" in row:
            r = row["hidden_frames_only"]
            print("{:<25} F1_D {:.4f} | recall LIVE {:.4f}".format(row["method"], r["F1_D"], r["recall_LIVE"]))
    np.savez_compressed(args.out / "validation_predictions.npz", frame=frame, half=half,
                        truth=truth, hidden=hidden, **pred_store)
    report = {
        "train_match": "DFL-MAT-J03WMX", "validation_match": "DFL-MAT-J03WN1",
        "config": CONFIG, "feature_names": NAMES, "team_radius": TEAM_RADIUS,
        "state_radius": STATE_RADIUS, "input_hashes": {"train": hashes_train, "valid": hashes_valid},
        "script_sha256": sha256(__file__), "model_sha256": sha256(model_path),
        "source": "DFL / Bassek et al. (2025), doi:10.6084/m9.figshare.28196177.v1, CC BY 4.0",
        "results": rows, "challenge_data_read": False, "challenge_gt_read": False,
        "validation_used_for_fitting_or_early_stopping": False,
        "stress_protocol": "Hide ball for seconds 10..15 of each 20-second cycle inside each half. No labels choose the gaps. This is not real occlusion validation.",
        "method": "Two binary LIVE/D models, with and without ball features. Symmetric side augmentation on training data. Team from fixed nearest-reference2s; carry across missing team evidence within each half. Centred probability average followed by fixed 0.5 threshold.",
        "limitations": "One training match and one development-validation match. No independent final test. External positions are cleaner and have persistent player identities; challenge ATD does not. No challenge performance claim.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nLaporan:", args.out / "report.json")
    print("Model:", model_path)
    print("Selesai: eksperimen EKSTERNAL, bukan submission atau skor challenge.")


if __name__ == "__main__":
    main()
