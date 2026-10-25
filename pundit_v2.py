"""Pundit Deculs v2: external development, frozen test, then local inference.

No challenge labels are read. Uses only publicly released IDSSE labels for
development. Existing v1 files remain unchanged. Run develop before test.
"""
import argparse
from collections import Counter
import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "4")
import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import ekstrak_fitur_idsse as reader
import latih_idsse_awal as v1
import eksperimen_urutan_idsse as temporal_v1

ROOT = Path(__file__).resolve().parent
DEV = ("J03WMX", "J03WN1", "J03WOH")
TEST = ("J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FILES = {
    "J03WMX": ((51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b"), (51643514, "e4273a2adf5dbd9d4ec42def2f01905c")),
    "J03WN1": ((51643472, "37b0f85588c0acfb995fe633c7ac2e0d"), (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2")),
    "J03WOH": ((51643478, "b9710781d400dd812f62b0a6e3d7bce8"), (51643520, "3efff25a86a61768e7d253b4bba137a3")),
    "J03WOY": ((51643481, "7b021e577fca917a63111d55cc29ae5b"), (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e")),
    "J03WPY": ((51643487, "65a4568e4f374af94d3c1015cbc7c4f5"), (51643526, "e56ff1336a334f8da51110fbe7cab764")),
    "J03WQQ": ((51643484, "5979209af316a914ce57347d0633df6c"), (51643529, "9105bad987191d3c3c362c1fce9a4fcd")),
    "J03WR9": ((51643490, "5e2d580d3d8ed97e32959de90d04992b"), (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d")),
}
STEP, HZ = 5, 5
SCENARIOS = ("clean", "ball_gaps", "mixed_tracking")
RECIPES = ("clean_only", "robust_aug")
STRENGTHS = (0.0, 0.35, 1.0)
CONFIG = dict(learning_rate=0.05, max_iter=140, max_leaf_nodes=15,
              min_samples_leaf=60, l2_regularization=3.0, class_weight="balanced",
              early_stopping=False, random_state=2027)
LABELS = np.array(["H", "A", "D"])


def sha(path):
    return v1.sha256(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_for(match, scenario, namespace):
    return int.from_bytes(hashlib.sha256((match + scenario + namespace).encode()).digest()[:4], "little")


def obtain(match, root):
    folder = root / match
    folder.mkdir(parents=True, exist_ok=True)
    for name, (fid, md5) in zip(("match_information.xml", "positions.xml"), FILES[match]):
        path = folder / name
        if path.exists() and reader.hash_file(path)["md5"] == md5:
            continue
        exe = shutil.which("curl.exe" if os.name == "nt" else "curl")
        if not exe:
            raise RuntimeError("curl tidak tersedia; unduh Figshare file " + str(fid))
        print("Unduh/melanjutkan", match, name, flush=True)
        subprocess.run([exe, "-L", "--fail", "--retry", "5", "--connect-timeout", "30",
                        "-C", "-", "-o", str(path), "https://ndownloader.figshare.com/files/" + str(fid)], check=True)
        if reader.hash_file(path)["md5"] != md5:
            raise ValueError("Hash sumber tidak cocok: " + str(path))
    return folder


def prepare_raw(match, root):
    folder = root / match
    cache = folder / "v2_cache/raw_5hz.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            if str(f["match"]) != match:
                raise ValueError("Identitas cache salah.")
            return {k: f[k].copy() for k in f.files}
    obtain(match, root)
    metadata = ET.parse(folder / "match_information.xml").getroot()
    g, env = metadata.find(".//General"), metadata.find(".//Environment")
    if g is None or g.get("MatchId") != "DFL-MAT-" + match:
        raise ValueError("Metadata pertandingan tidak sesuai.")
    home, away = g.get("HomeTeamId"), g.get("GuestTeamId") or g.get("AwayTeamId")
    data = reader.read_coordinates_and_targets(folder / "positions.xml", home, away, "DFL-MAT-" + match)
    result = {"match": np.array(match), "title": np.array(g.get("MatchTitle")),
              "pitch": np.array([float(env.get("PitchX")), float(env.get("PitchY"))])}
    for h, period in enumerate(("firstHalf", "secondHalf"), 1):
        d = data.pop(period)
        centers = np.minimum(np.arange(0, len(d["frame"]), STEP) + STEP // 2, len(d["frame"]) - 1)
        prefix = "h" + str(h) + "_"
        result[prefix + "home"] = d["teams"][home][centers]
        result[prefix + "away"] = d["teams"][away][centers]
        result[prefix + "ball"] = d["ball"][centers]
        result[prefix + "y25"] = d["target"]
        result[prefix + "y5"] = d["target"][centers]
        result[prefix + "frame25"] = d["frame"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_npz(cache, **result)
    return result


def canonical_cloud(xy):
    p = xy[np.isfinite(xy).all(axis=1)].astype(np.float64)
    return p[np.lexsort((p[:, 1], p[:, 0]))] if len(p) else p


def cloud_motion(previous, current, seconds):
    vectors = np.full_like(current, np.nan)
    if len(previous) and len(current):
        cost = np.linalg.norm(previous[:, None] - current[None, :], axis=-1)
        old, new = linear_sum_assignment(cost)
        vectors[new] = (current[new] - previous[old]) / seconds
        speeds = np.linalg.norm(vectors[new], axis=1)
        return vectors, float(np.median(speeds)), float(np.percentile(speeds, 75)), len(new) / len(current)
    return vectors, np.nan, np.nan, 0.0


def context_indices(known):
    n = len(known)
    last = np.maximum.accumulate(np.where(known, np.arange(n), -1))
    next_ = np.minimum.accumulate(np.where(known, np.arange(n), n)[::-1])[::-1]
    return last, next_


def rolling_mean(values, radius):
    ok = np.isfinite(values)
    sums = np.r_[0, np.cumsum(np.where(ok, values, 0), dtype=float)]
    counts = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(values))
    lo, hi = np.maximum(0, i - radius), np.minimum(len(values), i + radius + 1)
    num, den = sums[hi] - sums[lo], counts[hi] - counts[lo]
    return np.divide(num, den, out=np.full(len(values), np.nan), where=den > 0)


def features_v2(home, away, ball, pitch):
    """Identity-invariant input geometry; no targets/IDs/frame numbers accepted."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    last, next_ = context_indices(observed)
    index = np.arange(n)
    columns = {"ball_observed": observed.astype(float)}
    for lag, name in ((1, "02"), (5, "1s")):
        velocity = np.full((n, 2), np.nan)
        if n > lag:
            velocity[lag:] = (ball[lag:] - ball[:-lag]) * (HZ / lag)
        columns["ball_speed_" + name] = np.linalg.norm(velocity, axis=1)
        if lag == 1:
            ball_velocity = velocity
    columns["ball_edge_distance"] = np.minimum(pitch[0] / 2 - np.abs(ball[:, 0]), pitch[1] / 2 - np.abs(ball[:, 1]))
    columns["ball_gap_before_s"] = np.where(last >= 0, (index - last) / HZ, np.nan)
    columns["ball_gap_after_s"] = np.where(next_ < n, (next_ - index) / HZ, np.nan)
    columns["ball_observed_ratio_2s"] = rolling_mean(observed.astype(float), 5)
    centroids = {}
    for side, coords in (("home", home), ("away", away)):
        clouds = [canonical_cloud(x) for x in coords]
        fields = {k: np.full(n, np.nan) for k in (
            "count", "spread_median", "spread_p75", "nearest", "second_nearest", "within_2m", "within_5m",
            "motion02_median", "motion02_p75", "motion1s_median", "motion1s_p75",
            "matched02_ratio", "matched1s_ratio", "nearest_relative_speed", "last_ball_nearest", "next_ball_nearest")}
        centroid = np.full((n, 2), np.nan)
        for t, cloud in enumerate(clouds):
            fields["count"][t] = len(cloud)
            if not len(cloud):
                continue
            centroid[t] = cloud.mean(axis=0)
            spread = np.linalg.norm(cloud - centroid[t], axis=1)
            fields["spread_median"][t], fields["spread_p75"][t] = np.percentile(spread, [50, 75])
            nearest = None
            if observed[t]:
                distance = np.linalg.norm(cloud - ball[t], axis=1)
                nearest = int(np.argmin(distance))
                fields["nearest"][t] = distance[nearest]
                fields["second_nearest"][t] = np.partition(distance, 1)[1] if len(distance) > 1 else np.nan
                fields["within_2m"][t] = np.sum(distance <= 2)
                fields["within_5m"][t] = np.sum(distance <= 5)
            for lag, tag in ((1, "02"), (5, "1s")):
                if t < lag:
                    continue
                vectors, med, p75, ratio = cloud_motion(clouds[t - lag], cloud, lag / HZ)
                fields["motion" + tag + "_median"][t] = med
                fields["motion" + tag + "_p75"][t] = p75
                fields["matched" + tag + "_ratio"][t] = ratio
                if lag == 1 and nearest is not None:
                    fields["nearest_relative_speed"][t] = np.linalg.norm(vectors[nearest] - ball_velocity[t])
            if last[t] >= 0:
                fields["last_ball_nearest"][t] = np.linalg.norm(cloud - ball[last[t]], axis=1).min()
            if next_[t] < n:
                fields["next_ball_nearest"][t] = np.linalg.norm(cloud - ball[next_[t]], axis=1).min()
        for key, value in fields.items():
            columns[side + "_" + key] = value
        centroids[side] = centroid
    columns["centroid_separation"] = np.linalg.norm(centroids["home"] - centroids["away"], axis=1)
    columns["away_minus_home_nearest"] = columns["away_nearest"] - columns["home_nearest"]
    columns["abs_nearest_margin"] = np.abs(columns["away_minus_home_nearest"])
    h, a = columns["home_nearest"], columns["away_nearest"]
    known = np.isfinite(h) | np.isfinite(a)
    vote = np.full(n, np.nan)
    vote[known] = np.where(np.where(np.isfinite(h[known]), h[known], np.inf)
                           <= np.where(np.isfinite(a[known]), a[known], np.inf), 1.0, 0.0)
    vote[np.isfinite(h) & np.isfinite(a) & np.isclose(h, a, rtol=0, atol=1e-6)] = 0.5
    vl, vn = context_indices(known)
    columns["last_vote_home"] = np.where(vl >= 0, vote[np.maximum(vl, 0)], np.nan)
    columns["next_vote_home"] = np.where(vn < n, vote[np.minimum(vn, n - 1)], np.nan)
    columns["vote_home_1s"] = rolling_mean(vote, 5)
    columns["vote_home_2s"] = rolling_mean(vote, 10)
    for key in ("ball_speed_02", "home_nearest", "away_nearest", "home_motion02_median", "away_motion02_median"):
        columns[key + "_context1s"] = rolling_mean(columns[key], 5)
    names = tuple(columns)
    X = np.column_stack(list(columns.values())).astype(np.float32)
    if np.isinf(X).any():
        raise ValueError("Infinite feature value.")
    return X, names


def corrupt(home, away, ball, scenario, seed):
    h, a, b = home.copy(), away.copy(), ball.copy()
    rng = np.random.default_rng(seed)
    if scenario == "clean":
        return h, a, b, np.zeros(len(b), dtype=bool)
    hidden = np.zeros(len(b), dtype=bool)
    for t in range(1, len(b)):
        hidden[t] = rng.random() >= 0.15 if hidden[t - 1] else rng.random() < 0.10
    b[hidden] = np.nan
    if scenario == "mixed_tracking":
        # Label-blind correlated measurement noise, player dropout and duplicates.
        for xy, sigma in ((h, .15), (a, .15), (b, .35)):
            noise = rng.normal(0, sigma, size=xy.shape)
            for t in range(1, len(noise)):
                noise[t] = .7 * noise[t - 1] + np.sqrt(1 - .7**2) * noise[t]
            xy += noise
        teams = []
        for coords in (h, a):
            dropped = np.zeros(coords.shape[1], dtype=bool)
            for t in range(len(coords)):
                u = rng.random(len(dropped))
                dropped = np.where(dropped, u >= .25, u < .04)
                coords[t, dropped] = np.nan
            duplicates = np.full((len(coords), 2, 2), np.nan, dtype=np.float32)
            for t in range(len(coords)):
                seen = np.flatnonzero(np.isfinite(coords[t]).all(axis=1))
                if len(seen) and rng.random() < .15:
                    k = int(rng.integers(1, 3))
                    picked = rng.choice(seen, size=k, replace=True)
                    duplicates[t, :k] = coords[t, picked] + rng.normal(0, .15, size=(k, 2))
            combined = np.concatenate([coords, duplicates], axis=1)
            permutation = np.arange(combined.shape[1])
            for t in range(len(combined)):
                if t and t % 50 == 0:
                    pair = rng.choice(len(permutation), size=2, replace=False)
                    permutation[pair] = permutation[pair[::-1]]
                combined[t] = combined[t, permutation]
            teams.append(combined)
        h, a = teams
    return h, a, b, hidden


def feature_signature():
    text = "".join(inspect.getsource(f) for f in (canonical_cloud, cloud_motion, context_indices,
                    rolling_mean, features_v2, corrupt, mirror_features, transitions, decode))
    return hashlib.sha256((text + repr((STEP, HZ, SCENARIOS, CONFIG))).encode()).hexdigest()


def dataset(match, scenario, root, namespace):
    folder = root / match / "v2_cache"
    path = folder / (namespace + "_" + scenario + ".npz")
    sig = feature_signature()
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f["signature"]) == sig:
                return {k: f[k].copy() for k in f.files}
    raw = prepare_raw(match, root)
    Xs, old_Xs, y5, y25, half5, half25, hidden25, names = [], [], [], [], [], [], [], None
    for h in (1, 2):
        p = "h" + str(h) + "_"
        home, away, ball, hidden = corrupt(raw[p + "home"], raw[p + "away"], raw[p + "ball"], scenario,
                                         seed_for(match, scenario, namespace + str(h)))
        X, fn = features_v2(home, away, ball, raw["pitch"])
        # Fair v1 comparison: the same sampled/corrupted coordinates, held constant
        # over each 5-frame block, passed through the original 25 Hz v1 extractor.
        n25 = len(raw[p + "y25"])
        old_Xs.append(reader.make_features(np.repeat(home, STEP, axis=0)[:n25],
                                           np.repeat(away, STEP, axis=0)[:n25],
                                           np.repeat(ball, STEP, axis=0)[:n25]))
        if names is not None and names != fn:
            raise ValueError("Feature schema differs by half.")
        names = fn
        Xs.append(X); y5.append(raw[p + "y5"]); y25.append(raw[p + "y25"])
        half5.append(np.full(len(X), h, dtype=np.int8))
        half25.append(np.full(len(raw[p + "y25"]), h, dtype=np.int8))
        hidden25.append(np.repeat(hidden, STEP)[:len(raw[p + "y25"])])
    result = {"X": np.concatenate(Xs), "v1X25": np.concatenate(old_Xs), "y5": np.concatenate(y5), "y25": np.concatenate(y25),
              "half5": np.concatenate(half5), "half25": np.concatenate(half25),
              "hidden25": np.concatenate(hidden25), "names": np.array(names),
              "match": np.array(match), "signature": np.array(sig)}
    folder.mkdir(parents=True, exist_ok=True)
    save_npz(path, **result)
    print("Cache", match, scenario, result["X"].shape, flush=True)
    return result


def mirror_features(X, names):
    out = X.copy()
    for i, name in enumerate(names):
        if name.startswith("home_"):
            other = "away_" + name[5:]
            if other in names:
                j = names.index(other)
                out[:, i], out[:, j] = X[:, j], X[:, i]
        elif name in ("last_vote_home", "next_vote_home", "vote_home_1s", "vote_home_2s"):
            out[:, i] = 1 - X[:, i]
        elif name == "away_minus_home_nearest":
            out[:, i] = -X[:, i]
    return out


def transitions(sets):
    counts, initial = np.ones((3, 3)), np.ones(3)
    swap = np.array([1, 0, 2])
    for d in sets:
        for h in (1, 2):
            labels = d["y5"][d["half5"] == h]
            c = np.full(len(labels), -1, dtype=int)
            for i, k in enumerate(LABELS):
                c[labels == k] = i
            for seq in (c, np.where(c >= 0, swap[np.maximum(c, 0)], -1)):
                if len(seq) and seq[0] >= 0:
                    initial[seq[0]] += 1
                ok = (seq[:-1] >= 0) & (seq[1:] >= 0)
                np.add.at(counts, (seq[:-1][ok], seq[1:][ok]), 1)
    return counts / counts.sum(axis=1, keepdims=True), initial / initial.sum()


def decode(prob, transition, initial, strength):
    if strength == 0:
        return np.argmax(prob, axis=1)
    logp = np.log(np.clip(prob, 1e-6, 1))
    back = np.zeros((len(prob), 3), dtype=np.int8)
    score = logp[0] + strength * np.log(initial)
    cost = strength * np.log(transition)
    for t in range(1, len(prob)):
        paths = score[:, None] + cost
        back[t] = paths.argmax(axis=0)
        score = paths[back[t], np.arange(3)] + logp[t]
    seq = np.empty(len(prob), dtype=np.int8); seq[-1] = score.argmax()
    for t in range(len(prob) - 1, 0, -1):
        seq[t - 1] = back[t, seq[t]]
    return seq


def fit(sets, recipe):
    selected = sets if recipe == "robust_aug" else [d for d in sets if str(d["scenario"]) == "clean"]
    names = tuple(selected[0]["names"].tolist())
    X = np.concatenate([d["X"] for d in selected])
    labels = np.concatenate([d["y5"] for d in selected])
    ok = np.isin(labels, LABELS)
    X, labels = X[ok], labels[ok]
    y = np.array([int(np.flatnonzero(LABELS == s)[0]) for s in labels], dtype=np.int8)
    inputs = np.concatenate([X, mirror_features(X, names)])
    target = np.r_[y, np.array([1, 0, 2])[y]]
    model = HistGradientBoostingClassifier(**CONFIG).fit(inputs, target)
    return model, names


def probabilities(model, d):
    p = model.predict_proba(d["X"])
    return p[:, [list(model.classes_).index(k) for k in range(3)]]


def predict25(prob, d, transition, initial, strength):
    result = np.full(len(d["y25"]), "X", dtype="<U1")
    names = tuple(d["names"].tolist())
    for h in (1, 2):
        m5, m25 = d["half5"] == h, d["half25"] == h
        seq = LABELS[decode(prob[m5], transition, initial, strength)]
        x = d["X"][m5]
        no_observations = (x[:, names.index("ball_observed")] == 0) & (
            x[:, names.index("home_count")] + x[:, names.index("away_count")] == 0)
        seq[no_observations] = "X"
        result[m25] = np.repeat(seq, STEP)[:int(m25.sum())]
    return result


def stats(d, pred):
    r = v1.summary(d["y25"], pred, d["half25"])
    hidden_live = d["hidden25"] & np.isin(d["y25"], ["H", "A"])
    r["hidden_live_team_accuracy"] = float(np.mean(pred[hidden_live] == d["y25"][hidden_live])) if hidden_live.any() else None
    return r


def develop(root, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen.json").exists():
        print("Model v2 sudah dibekukan:", out / "frozen.json")
        return
    protocol = {"development": DEV, "reserved_test": TEST, "scenarios": SCENARIOS,
                "recipes": RECIPES, "decoder_strengths": STRENGTHS, "model_config": CONFIG,
                "selection": "LOMO mean spellF1, equal match/scenario weights; macroF1 tie-break",
                "signature": feature_signature(), "challenge_labels_read": False}
    save_json(out / "protocol.json", protocol)
    cases = {}
    for match in DEV:
        for scenario in SCENARIOS:
            d = dataset(match, scenario, root, "development")
            d["scenario"] = np.array(scenario)
            cases[(match, scenario)] = d
    rows = []
    with threadpool_limits(limits=4):
        for held in DEV:
            train_cases = [d for (m, _), d in cases.items() if m != held]
            clean_train = [cases[(m, "clean")] for m in DEV if m != held]
            transition, initial = transitions(clean_train)
            for recipe in RECIPES:
                print("DEVELOPMENT fold", held, "recipe", recipe, flush=True)
                model, _ = fit(train_cases, recipe)
                for scenario in SCENARIOS:
                    d = cases[(held, scenario)]
                    p = probabilities(model, d)
                    for strength in STRENGTHS:
                        pred = predict25(p, d, transition, initial, strength)
                        r = stats(d, pred)
                        rows.append({"match": held, "scenario": scenario, "recipe": recipe, "strength": strength, **r})
                save_json(out / "development_rows.json", rows)
    scores = []
    for recipe in RECIPES:
        for strength in STRENGTHS:
            group = [r for r in rows if r["recipe"] == recipe and r["strength"] == strength]
            scores.append({"recipe": recipe, "strength": strength,
                           "mean_spellF1": float(np.mean([r["spellF1"] for r in group])),
                           "mean_macroF1": float(np.mean([r["macroF1"] for r in group]))})
    scores.sort(key=lambda r: (-r["mean_spellF1"], -r["mean_macroF1"], r["recipe"], r["strength"]))
    print("\nDEVELOPMENT - mean across held-out development games and scenarios")
    for r in scores:
        print(r)
    selected = scores[0]
    print("Refit pada tiga pertandingan pengembangan:", selected, flush=True)
    with threadpool_limits(limits=4):
        model, names = fit(list(cases.values()), selected["recipe"])
    transition, initial = transitions([cases[(m, "clean")] for m in DEV])
    model_path = out / "model.joblib"
    temporary_model = out / "model.joblib.tmp"
    joblib.dump({"model": model, "names": names, "transition": transition, "initial": initial,
                 "strength": selected["strength"], "signature": feature_signature()}, temporary_model)
    temporary_model.replace(model_path)
    frozen = {"development_matches": DEV, "reserved_matches": TEST, "selected": selected,
              "selection_table": scores, "signature": feature_signature(), "source_sha256": sha(__file__),
              "model_sha256": sha(model_path), "challenge_gt_read": False,
              "versions": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
              "external_source": "DFL / Bassek et al. 2025; doi:10.6084/m9.figshare.28196177.v1; CC BY 4.0"}
    save_json(out / "frozen.json", frozen)
    print("MODEL V2 BEKU:", out / "frozen.json", flush=True)


def test(root, out, v1_model_dir, v1_temporal_dir):
    frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
    if frozen["signature"] != feature_signature() or sha(out / "model.joblib") != frozen["model_sha256"]:
        raise ValueError("Model/fitur berubah setelah pembekuan.")
    if (out / "test_report.json").exists():
        print("Pemeriksaan akhir sudah pernah dijalankan; lihat", out / "test_report.json")
        return
    payload = joblib.load(out / "model.joblib")
    old_report = json.loads((v1_model_dir / "report.json").read_text(encoding="utf-8"))
    if sha(v1_model_dir / "models.joblib") != old_report["model_sha256"]:
        raise ValueError("Bobot v1 tidak cocok dengan laporan.")
    old_models = joblib.load(v1_model_dir / "models.joblib")["models"]
    with np.load(v1_temporal_dir / "sequence_parameters.npz", allow_pickle=False) as f:
        old_transition, old_initial = f["transition"].copy(), f["initial"].copy()
    save_json(out / "test_frozen_inputs.json", {
        "v2_model_sha256": sha(out / "model.joblib"),
        "v1_model_sha256": sha(v1_model_dir / "models.joblib"),
        "v1_sequence_sha256": sha(v1_temporal_dir / "sequence_parameters.npz"),
        "feature_signature": feature_signature(), "reserved_matches": TEST})
    rows = []
    print("EMPAT MATCH CADANGAN: model/konfigurasi tidak berubah.", flush=True)
    with threadpool_limits(limits=4):
        for match in TEST:
            for scenario in SCENARIOS:
                d = dataset(match, scenario, root, "reserved_test")
                p = probabilities(payload["model"], d)
                pred = predict25(p, d, payload["transition"], payload["initial"], payload["strength"])
                baseline = temporal_v1.predict_sequence(old_models, d["v1X25"], d["half25"], old_transition, old_initial)
                reference = v1.nearest_reference(d["v1X25"], d["half25"])
                for method, result in (("nearest_reference2s", reference), ("frozen_v1", baseline), ("selected_v2", pred)):
                    r = {"match": match, "scenario": scenario, "method": method, **stats(d, result)}
                    rows.append(r)
                    print("TEST", match, scenario, method, "spellF1", round(r["spellF1"], 4),
                          "macroF1", round(r["macroF1"], 4), "F1_D", round(r["F1_D"], 4), flush=True)
                save_npz(out / (match + "_" + scenario + "_prediction.npz"), v2=pred, v1=baseline)
                save_json(out / "test_progress.json", rows)
    summary = {s: {method: {k: float(np.mean([r[k] for r in rows if r["scenario"] == s and r["method"] == method]))
                            for k in ("spellF1", "macroF1", "F1_D")}
                   for method in ("nearest_reference2s", "frozen_v1", "selected_v2")} for s in SCENARIOS}
    report = {"frozen": frozen, "results": rows, "mean_by_scenario": summary,
              "challenge_gt_read": False, "test_used_for_tuning": False,
              "limitations": "Four external games only; synthetic corruption is not real challenge ground truth. No automatic post-test model changes. v1 receives the same 5 Hz observations repeated to 25 Hz, not its original full-rate input."}
    save_json(out / "test_report.json", report)
    print("\nMEAN EMPAT PERTANDINGAN:", json.dumps(summary, indent=2))
    print("Laporan:", out / "test_report.json")


def read_atd(path_meta, path_raw, n, pitch):
    root = ET.parse(path_meta).getroot()
    teams = list(root.iter("Team"))
    ids = [t.get("id") for t in teams]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Metadata ATD harus memiliki dua tim.")
    track_teams = [p.get("teamId") for p in root.iter("Player")]
    if not track_teams or set(track_teams) - set(ids):
        raise ValueError("Slot ATD tidak memiliki tim yang benar.")
    side = np.array([ids.index(t) for t in track_teams])
    players = np.full((n, len(side), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    scale = np.asarray(pitch)
    count = 0
    with path_raw.open(encoding="utf-8-sig") as f:
        for row, line in enumerate(f):
            parts = line.strip().split(":")
            if row >= n or len(parts) != 3 or int(parts[0]) != row + 1:
                raise ValueError("Indeks/struktur ATD tidak sesuai.")
            groups = np.array([p.split(",") for p in parts[1].split(";")], dtype=float)
            b = np.asarray(parts[2].split(","), dtype=float)
            if groups.shape != (len(side), 3) or b.shape != (3,):
                raise ValueError("Jumlah slot/kolom ATD tidak sesuai.")
            xy = groups[:, :2] * scale - scale / 2
            xy[~np.isfinite(xy).all(axis=1)] = np.nan
            players[row] = xy
            if np.isfinite(b[:2]).all():
                ball[row] = b[:2] * scale - scale / 2
            count += 1
    if count != n or not np.any(side == 0) or not np.any(side == 1):
        raise ValueError("Jumlah frame/tim ATD tidak sesuai.")
    return players[:, side == 0], players[:, side == 1], ball, {
        "H": teams[0].findtext("Name"), "A": teams[1].findtext("Name")}


def diagnose_long_gaps(root, out, v1_model_dir, v1_temporal_dir):
    """Additional fixed stress diagnostics; never used to refit/select v2."""
    frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
    if frozen["signature"] != feature_signature() or sha(out / "model.joblib") != frozen["model_sha256"]:
        raise ValueError("Model/fitur berubah setelah pembekuan.")
    if (out / "long_gap_report.json").exists():
        print("Diagnostik celah panjang sudah tersedia:", out / "long_gap_report.json")
        return
    payload = joblib.load(out / "model.joblib")
    old_models = joblib.load(v1_model_dir / "models.joblib")["models"]
    with np.load(v1_temporal_dir / "sequence_parameters.npz", allow_pickle=False) as f:
        ot, oi = f["transition"], f["initial"]
    protocol = {"matches": TEST, "model_sha256": sha(out / "model.joblib"),
                "cases": {"fixed5s": "seconds 10..15 in each 20s cycle",
                          "fixed15s": "seconds 10..25 in each 40s cycle"},
                "use": "Additional diagnostics only; no refitting or configuration changes."}
    save_json(out / "long_gap_protocol.json", protocol)
    rows = []
    with threadpool_limits(limits=4):
        for match in TEST:
            raw = prepare_raw(match, root)
            for scenario, cycle, end in (("fixed5s", 20, 15), ("fixed15s", 40, 25)):
                xs, oldxs, ys, half5, half25, hidden25 = [], [], [], [], [], []
                for h in (1, 2):
                    p = "h" + str(h) + "_"
                    home, away, ball = raw[p + "home"], raw[p + "away"], raw[p + "ball"].copy()
                    seconds = np.arange(len(ball)) / HZ
                    hidden = (seconds % cycle >= 10) & (seconds % cycle < end)
                    ball[hidden] = np.nan
                    X, names = features_v2(home, away, ball, raw["pitch"])
                    n25 = len(raw[p + "y25"])
                    oldxs.append(reader.make_features(np.repeat(home, STEP, axis=0)[:n25],
                                                      np.repeat(away, STEP, axis=0)[:n25],
                                                      np.repeat(ball, STEP, axis=0)[:n25]))
                    xs.append(X); ys.append(raw[p + "y25"])
                    half5.append(np.full(len(X), h, dtype=np.int8))
                    half25.append(np.full(n25, h, dtype=np.int8))
                    hidden25.append(np.repeat(hidden, STEP)[:n25])
                d = {"X": np.concatenate(xs), "names": np.array(names), "y25": np.concatenate(ys),
                     "half5": np.concatenate(half5), "half25": np.concatenate(half25),
                     "hidden25": np.concatenate(hidden25)}
                prob = probabilities(payload["model"], d)
                pred = predict25(prob, d, payload["transition"], payload["initial"], payload["strength"])
                old = temporal_v1.predict_sequence(old_models, np.concatenate(oldxs), d["half25"], ot, oi)
                for method, result in (("frozen_v1", old), ("selected_v2", pred)):
                    r = {"match": match, "scenario": scenario, "method": method, **stats(d, result)}
                    rows.append(r)
                    print("LONG GAP", match, scenario, method, "spellF1", round(r["spellF1"], 4),
                          "team-hidden", round(r["hidden_live_team_accuracy"], 4), flush=True)
                save_json(out / "long_gap_progress.json", rows)
    summary = {s: {m: {k: float(np.mean([r[k] for r in rows if r["scenario"] == s and r["method"] == m]))
                       for k in ("spellF1", "macroF1", "F1_D", "hidden_live_team_accuracy")}
                   for m in ("frozen_v1", "selected_v2")} for s in ("fixed5s", "fixed15s")}
    save_json(out / "long_gap_report.json", {"protocol": protocol, "results": rows, "mean_by_scenario": summary})
    print("\nDIAGNOSTIK CELAH PANJANG:", json.dumps(summary, indent=2), flush=True)


def candidate(data_dir, out, destination):
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
    if feature_signature() != frozen["signature"] or sha(out / "model.joblib") != frozen["model_sha256"]:
        raise ValueError("Model/fitur v2 tidak sesuai dengan pembekuan.")
    if not (out / "test_report.json").exists():
        raise ValueError("Selesaikan pemeriksaan empat pertandingan cadangan terlebih dahulu.")
    payload = joblib.load(out / "model.joblib")
    meta, raw = map(Path, require(["atd_meta", "atd_raw"], data_dir=str(data_dir)))
    n, pitch = source_frames(), np.asarray(GAMES[GAME]["pitch"])
    print("Inferensi ATD dengan model v2 beku; tanpa GT atau training.", flush=True)
    home, away, ball, teams = read_atd(meta, raw, n, pitch)
    Xs, half5, half25, video_frames, profiles = [], [], [], [], []
    for h, first, last in half_video_blocks(GAME):
        indices = np.minimum(np.arange(first, last + 1, STEP) + STEP // 2, last)
        X, names = features_v2(home[indices], away[indices], ball[indices], pitch)
        if tuple(names) != tuple(payload["names"]):
            raise ValueError("Schema fitur ATD berbeda dari schema latihan.")
        Xs.append(X)
        half5.append(np.full(len(X), h, dtype=np.int8))
        half25.append(np.full(last - first + 1, h, dtype=np.int8))
        video_frames.append(np.arange(first, last + 1))
        profiles.append({"half": h, "frames": last - first + 1,
                         "sampled_ball_missing_pct": 100 * float(np.mean(X[:, names.index("ball_observed")] == 0)),
                         "median_features": {k: float(np.nanmedian(X[:, names.index(k)])) for k in (
                             "home_count", "away_count", "home_motion02_median", "away_motion02_median",
                             "ball_speed_1s", "home_nearest", "away_nearest")}})
        print("Fitur babak", h, "selesai:", X.shape, flush=True)
    frames = np.concatenate(video_frames)
    d = {"X": np.concatenate(Xs), "names": np.array(names),
         "half5": np.concatenate(half5), "half25": np.concatenate(half25),
         "y25": np.full(len(frames), "X", dtype="<U1")}
    with threadpool_limits(limits=4):
        prob = probabilities(payload["model"], d)
        pred = predict25(prob, d, payload["transition"], payload["initial"], payload["strength"])
    full = np.full(n, "X", dtype="<U1"); full[frames] = pred
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "Pundit_Deculs.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, x) for i, x in enumerate(full))
    reason = v1.pm.validate_submission(v1.pm.read_label_csv(path), n)
    if reason:
        raise ValueError(reason)
    for profile in profiles:
        mask = d["half25"] == profile["half"]
        profile["labels"] = dict(Counter(pred[mask].tolist()))
    report = {"method": "Pundit v2 identity-invariant geometry + supervised H/A/D + frozen temporal decoder",
              "model_sha256": sha(out / "model.joblib"), "signature": feature_signature(),
              "source_sha256": sha(__file__), "candidate_sha256": sha(path),
              "input_hashes": {"atd_meta": sha(meta), "atd_raw": sha(raw)},
              "rows": n, "teams": teams, "label_counts": dict(Counter(full.tolist())),
              "profiles": profiles, "selected": frozen["selected"],
              "development_matches": DEV, "external_check_matches": TEST,
              "versions": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
              "training_performed": False, "challenge_gt_read": False,
              "sampling": "Centre of each 5-frame block, 5 Hz; predictions repeated to the 25 Hz reference timeline. Each half is independent.",
              "limits": "External labels/clean tracking and synthetic corruption do not establish challenge accuracy. Point-cloud motion is a minimum-assignment displacement statistic, not true player velocity or a referee detector."}
    save_json(destination / "report.json", report)
    for profile in profiles:
        print("AUDIT babak", profile["half"], profile)
    print("FORMAT: OK |", n, "baris")
    print("Label:", report["label_counts"])
    print("SHA256:", report["candidate_sha256"])
    print("CSV:", path)
    print("Audit:", destination / "report.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("develop", "test", "diagnose", "candidate", "all"))
    p.add_argument("--data-root", type=Path, default=ROOT / "data/external/idsse")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation/experiments/idsse_v2")
    p.add_argument("--v1-model-dir", type=Path, default=ROOT / "evaluation/experiments/idsse_external_v1")
    p.add_argument("--v1-temporal-dir", type=Path, default=ROOT / "evaluation/experiments/idsse_temporal_v1")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--candidate-out", type=Path, default=ROOT / "evaluation/candidates/idsse_v2")
    a = p.parse_args()
    if a.mode in ("candidate", "all") and a.data_dir is None:
        p.error("candidate/all memerlukan --data-dir")
    start = time.monotonic()
    if a.mode == "develop":
        develop(a.data_root, a.out)
    elif a.mode == "test":
        test(a.data_root, a.out, a.v1_model_dir, a.v1_temporal_dir)
    elif a.mode == "candidate":
        candidate(a.data_dir, a.out, a.candidate_out)
    elif a.mode == "diagnose":
        diagnose_long_gaps(a.data_root, a.out, a.v1_model_dir, a.v1_temporal_dir)
    else:
        develop(a.data_root, a.out)
        test(a.data_root, a.out, a.v1_model_dir, a.v1_temporal_dir)
        diagnose_long_gaps(a.data_root, a.out, a.v1_model_dir, a.v1_temporal_dir)
        candidate(a.data_dir, a.out, a.candidate_out)
    print("Waktu sesi {:.1f} menit".format((time.monotonic() - start) / 60))


if __name__ == "__main__":
    main()
