"""Uji urutan H/A/D pada IDSSE eksternal; tidak membaca challenge atau GT-nya.

Head LIVE/D tidak dilatih ulang. Probabilitas perpindahan keadaan dipelajari
hanya dari J03WMX. J03WN1 tetap validasi pengembangan, bukan final test.
Memerlukan latih_idsse_awal.py dan model hasilnya di project yang sama.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import latih_idsse_awal as base

LABELS = np.array(["H", "A", "D"])
STEP = 5  # 5 Hz: one model state per 200 ms, fixed before validation.
EPS = 1e-6


def block_targets(labels):
    """Training-only majority target for each 200 ms block; X blocks remain X."""
    out = []
    for first in range(0, len(labels), STEP):
        block = labels[first:first + STEP]
        count = np.array([np.sum(block == name) for name in LABELS])
        out.append(int(np.argmax(count)) if count.sum() else -1)
    return np.asarray(out, dtype=int)


def learn_transitions(train_target, half):
    counts = np.ones((3, 3), dtype=np.float64)  # Laplace pseudocount.
    starts = np.ones(3, dtype=np.float64)
    swap = np.array([1, 0, 2])
    for h in (1, 2):
        codes = block_targets(train_target[half == h])
        for seq in (codes, np.where(codes >= 0, swap[np.maximum(codes, 0)], -1)):
            if len(seq) and seq[0] >= 0:
                starts[seq[0]] += 1
            valid = (seq[:-1] >= 0) & (seq[1:] >= 0)
            np.add.at(counts, (seq[:-1][valid], seq[1:][valid]), 1)
    return counts / counts.sum(axis=1, keepdims=True), starts / starts.sum(), counts


def evidence(models, X, half):
    """No target enters the observations used by the sequence decoder."""
    k = list(models["full"].classes_).index(1)
    kp = list(models["players"].classes_).index(1)
    full = models["full"].predict_proba(X)[:, k]
    players = models["players"].predict_proba(X[:, 7:])[:, kp]
    dead = np.where(X[:, 0] == 0, players, full)
    dh, da = X[:, 2], X[:, 3]
    available = np.isfinite(dh) | np.isfinite(da)
    raw_home = available & (np.where(np.isfinite(dh), dh, np.inf)
                            <= np.where(np.isfinite(da), da, np.inf))
    raw_away = available & ~raw_home
    home = np.full(len(X), 0.5)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        hv = base.centred_mean(raw_home[idx].astype(float), base.TEAM_RADIUS)
        av = base.centred_mean(raw_away[idx].astype(float), base.TEAM_RADIUS)
        home[idx] = np.divide(hv, hv + av, out=np.full(len(idx), 0.5), where=(hv + av) > 0)
    dead, home = np.clip(dead, EPS, 1 - EPS), np.clip(home, EPS, 1 - EPS)
    # The binary classifier used balanced class weights. H and A share the LIVE
    # evidence equally in the absence of team information: (0.5,0.5,0.5) then
    # gives equal state likelihoods. These are relative likelihoods, not a
    # normalized three-class posterior or a claim of probability calibration.
    return np.column_stack(((1 - dead) * 2 * home,
                            (1 - dead) * 2 * (1 - home), dead))


def decode_half(likelihood, transition, initial):
    log_emissions = np.log(np.clip(likelihood, EPS, None))
    block_log = np.array([log_emissions[i:i + STEP].mean(axis=0)
                          for i in range(0, len(log_emissions), STEP)])
    if not len(block_log):
        return np.empty(0, dtype="<U1")
    back = np.zeros((len(block_log), 3), dtype=np.int8)
    score = np.log(initial) + block_log[0]
    log_transition = np.log(transition)
    for t in range(1, len(block_log)):
        choices = score[:, None] + log_transition
        back[t] = np.argmax(choices, axis=0)
        score = choices[back[t], np.arange(3)] + block_log[t]
    path = np.empty(len(block_log), dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(len(block_log) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return LABELS[np.repeat(path, STEP)[:len(likelihood)]]


def predict_sequence(models, X, half, transition, initial):
    likelihood = evidence(models, X, half)
    pred = np.full(len(X), "X", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        pred[idx] = decode_half(likelihood[idx], transition, initial)
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = base.ROOT / "data/external/idsse"
    parser.add_argument("--train-dir", type=Path, default=data / "J03WMX/features_v1")
    parser.add_argument("--valid-dir", type=Path, default=data / "J03WN1/features_v1")
    parser.add_argument("--model-dir", type=Path,
                        default=base.ROOT / "evaluation/experiments/idsse_external_v1")
    parser.add_argument("--out", type=Path,
                        default=base.ROOT / "evaluation/experiments/idsse_temporal_v1")
    args = parser.parse_args()
    _, ytrain, _, half_train, train_hashes = base.load_data(args.train_dir, "DFL-MAT-J03WMX")
    source_report = json.loads((args.model_dir / "report.json").read_text(encoding="utf-8"))
    model_path = args.model_dir / "models.joblib"
    if source_report["input_hashes"]["train"] != train_hashes:
        raise ValueError("Data latih berbeda dari data yang menghasilkan model.")
    if source_report["model_sha256"] != base.sha256(model_path):
        raise ValueError("Hash model tidak cocok dengan laporan pelatihannya.")
    payload = joblib.load(model_path)
    if payload.get("train_match") != "DFL-MAT-J03WMX" or tuple(payload["feature_names"]) != base.NAMES:
        raise ValueError("Sumber model/schema berbeda.")
    models = payload["models"]
    transition, initial, counts = learn_transitions(ytrain, half_train)
    print("Head LIVE/D: model eksternal tersimpan; tidak dilatih ulang.")
    print("Transisi dipelajari dari J03WMX saja; langkah waktu 200 ms.")
    print("Urutan baris/kolom: H, A, D")
    print(np.round(transition, 5))
    # No validation target affects transitions or prediction functions.
    X, truth, frame, half, valid_hashes = base.load_data(args.valid_dir, "DFL-MAT-J03WN1")
    masked, hidden = base.hide_ball(X, half)
    rows, saved = [], {}
    with base.threadpool_limits(limits=4):
        for scenario, inputs in (("bola_tersedia", X), ("bola_disembunyikan_5s", masked)):
            original = base.predictions(models, inputs, half)["external_full"]
            sequence = predict_sequence(models, inputs, half, transition, initial)
            print("\n" + scenario)
            print("method                 macroF1 spellF1   F1_D  poss false_D missed_D")
            for name, pred in (("external_full_awal", original), ("external_markov", sequence)):
                stats = base.summary(truth, pred, half)
                row = {"scenario": scenario, "method": name, **stats}
                if scenario == "bola_disembunyikan_5s":
                    row["hidden_frames_only"] = base.missing_summary(truth, pred, hidden)
                rows.append(row)
                saved[scenario + "__" + name] = pred
                print("{:<22} {:.4f}  {:.4f} {:.4f} {:>5} {:>7} {:>8}".format(
                    name, stats["macroF1"], stats["spellF1"], stats["F1_D"],
                    stats["possessions"], stats["false_D"], stats["missed_D"]))
    print("\nUJI HANYA FRAME BOLA DISEMBUNYIKAN (sintetis)")
    for r in rows:
        if "hidden_frames_only" in r:
            m = r["hidden_frames_only"]
            print("{:<22} F1_D {:.4f} | recall LIVE {:.4f}".format(r["method"], m["F1_D"], m["recall_LIVE"]))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "sequence_parameters.npz", transition=transition, initial=initial)
    np.savez_compressed(args.out / "validation_predictions.npz", frame=frame, half=half,
                        truth=truth, hidden=hidden, **saved)
    report = {
        "train_match": "DFL-MAT-J03WMX", "validation_match": "DFL-MAT-J03WN1",
        "input_hashes": {"train": train_hashes, "valid": valid_hashes},
        "model_sha256": base.sha256(model_path), "script_sha256": base.sha256(__file__),
        "training_script_sha256": base.sha256(base.__file__),
        "step_frames": STEP, "team_radius": base.TEAM_RADIUS,
        "transition_counts_with_pseudocount": counts.tolist(),
        "transition": transition.tolist(), "initial": initial.tolist(), "results": rows,
        "method": "Three-state Viterbi decoder. Transition and initial probabilities from external training labels plus home/away mirroring and Laplace pseudocounts. Relative emissions from frozen binary LIVE/D heads and nearest-team vote evidence. Mean log likelihood per 200 ms block. No grid search or calibrated posterior claim.",
        "challenge_data_read": False, "challenge_gt_read": False,
        "source": "DFL / Bassek et al. (2025), doi:10.6084/m9.figshare.28196177.v1, CC BY 4.0",
        "limitations": "Development validation on J03WN1 has influenced the choice to test a temporal method. Not an untouched final test. Synthetic ball hiding is not actual occlusion; clean tracking has persistent player identities unlike challenge ATD.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nLaporan:", args.out / "report.json")
    print("Kandidat challenge tidak dibuat atau diubah. Masih eksperimen eksternal.")


if __name__ == "__main__":
    main()
