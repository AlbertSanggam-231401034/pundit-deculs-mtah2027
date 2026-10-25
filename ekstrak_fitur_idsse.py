"""Ekstrak fitur IDSSE tanpa training, GT challenge, atau model lama.

Data: DFL; Bassek et al. (2025), doi:10.6084/m9.figshare.28196177.v1,
CC BY 4.0. Kecepatan dihitung dari koordinat (meter), bukan atribut S XML.
Label hanya dibaca untuk targets.npz. Fungsi fitur tidak menerima label.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
LAG = 25  # One second, fixed physical time scale; not chosen on challenge GT.
MATCHES = {
    "J03WMX": {
        "match_information.xml": "bc5d5b21dbcf36aa30f4cd0428309f6b",
        "positions.xml": "e4273a2adf5dbd9d4ec42def2f01905c",
    },
    "J03WN1": {
        "match_information.xml": "37b0f85588c0acfb995fe633c7ac2e0d",
        "positions.xml": "bacb2f609c1d1b0bfc906f6b9d4109f2",
    },
}
FEATURE_NAMES = (
    "ball_observed", "ball_displacement_speed_1s_mps",
    "nearest_home_m", "nearest_away_m", "nearest_any_m",
    "away_minus_home_distance_m", "abs_distance_margin_m",
    "home_visible", "away_visible",
    "players_displacement_speed_1s_median_mps",
    "players_displacement_speed_1s_p75_mps", "players_motion_samples",
    "home_spread_median_m", "away_spread_median_m", "centroid_separation_m",
)
PLAYER_ONLY_NAMES = FEATURE_NAMES[7:]


def hash_file(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def frames(path):
    """Yield frame-set metadata and frame attributes, without retaining XML nodes."""
    stack, context = [], None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag.rsplit("}", 1)[-1]
        if event == "start":
            stack.append(elem)
            if tag == "FrameSet":
                context = dict(elem.attrib)
            continue
        if tag == "Frame" and context is not None:
            yield context, dict(elem.attrib)
        if tag == "FrameSet":
            context = None
        stack.pop()
        if stack:
            stack[-1].remove(elem)
        elem.clear()


def target(status, owner):
    if status == "0":
        return "D"
    if status == "1":
        return {"1": "H", "2": "A"}.get(owner, "X")
    return "X"


def xy(attrs):
    try:
        pair = [float(attrs["X"]), float(attrs["Y"])]
    except (KeyError, ValueError):
        return [np.nan, np.nan]
    return pair if np.isfinite(pair).all() else [np.nan, np.nan]


def read_coordinates_and_targets(path, home_id, away_id, match_id="DFL-MAT-J03WMX"):
    slots = defaultdict(lambda: {home_id: set(), away_id: set()})
    targets = defaultdict(dict)
    clocks = defaultdict(lambda: [None, None])
    print("Lintasan 1/2: memeriksa indeks dan membaca target eksternal...", flush=True)
    for context, attrs in frames(path):
        if context.get("MatchId") != match_id:
            raise ValueError("MatchId tidak sesuai.")
        period, team = context["GameSection"], context["TeamId"]
        stamp = datetime.fromisoformat(attrs["T"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Timestamp tidak menyertakan zona waktu.")
        offset = round(stamp.timestamp() * 1_000_000) - int(attrs["N"]) * 40000
        bounds = clocks[period]
        bounds[0] = offset if bounds[0] is None else min(bounds[0], offset)
        bounds[1] = offset if bounds[1] is None else max(bounds[1], offset)
        if team.upper() == "BALL":
            n = int(attrs["N"])
            if n in targets[period]:
                raise ValueError("Frame bola duplikat.")
            targets[period][n] = target(attrs.get("BallStatus"), attrs.get("BallPossession"))
        elif team in (home_id, away_id):
            slots[period][team].add(context["PersonId"])
    if set(targets) != {"firstHalf", "secondHalf"}:
        raise ValueError("Babak bola tidak lengkap.")
    for period, (low, high) in clocks.items():
        if high - low > 2:
            raise ValueError("Acuan waktu antarobjek/25 Hz berbeda: " + period)
    print("Pemeriksaan waktu: konsisten 25 Hz di kedua babak.", flush=True)
    data = {}
    for period in ("firstHalf", "secondHalf"):
        ids = np.array(sorted(targets[period]), dtype=np.int64)
        if not np.array_equal(ids, np.arange(ids[0], ids[-1] + 1)):
            raise ValueError("Indeks bola terputus; audit dahulu sebelum ekstraksi.")
        maps = {team: {p: i for i, p in enumerate(sorted(slots[period][team]))}
                for team in (home_id, away_id)}
        if any(not m for m in maps.values()):
            raise ValueError("Salah satu tim tidak memiliki track.")
        data[period] = {
            "frame": ids, "target": np.array([targets[period][int(n)] for n in ids]),
            "maps": maps, "ball": np.full((len(ids), 2), np.nan, dtype=np.float32),
            "teams": {t: np.full((len(ids), len(maps[t]), 2), np.nan, dtype=np.float32)
                      for t in (home_id, away_id)},
        }
    del targets
    print("Lintasan 2/2: membaca koordinat pemain dan bola saja...", flush=True)
    for context, attrs in frames(path):
        period, team = context["GameSection"], context["TeamId"]
        if period not in data:
            continue
        d = data[period]
        n = int(attrs["N"]) - int(d["frame"][0])
        if not 0 <= n < len(d["frame"]):
            raise ValueError("Frame objek di luar rentang bola.")
        if team.upper() == "BALL":
            d["ball"][n] = xy(attrs)
        elif team in (home_id, away_id):
            slot = d["maps"][team][context["PersonId"]]
            if np.isfinite(d["teams"][team][n, slot]).any():
                raise ValueError("Koordinat pemain/frame duplikat.")
            d["teams"][team][n, slot] = xy(attrs)
    return data


def displacement_speed(coords, lag=LAG, fps=FPS):
    """Displacement/elapsed time, not travelled-path length; no cross-half pairs."""
    out = np.full(coords.shape[:-1], np.nan, dtype=np.float32)
    if len(coords) > lag:
        out[lag:] = np.linalg.norm(coords[lag:] - coords[:-lag], axis=-1) * (fps / lag)
    return out


def make_features(home, away, ball):
    """Only coordinates enter this function. No target/status/owner metadata."""
    h_seen = np.isfinite(home).all(axis=-1)
    a_seen = np.isfinite(away).all(axis=-1)
    b_seen = np.isfinite(ball).all(axis=-1)
    # All-NaN at track edges is expected and preserved for later train-only imputation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nh = np.nanmin(np.linalg.norm(home - ball[:, None, :], axis=-1), axis=1)
        na = np.nanmin(np.linalg.norm(away - ball[:, None, :], axis=-1), axis=1)
        hc, ac = np.nanmean(home, axis=1), np.nanmean(away, axis=1)
        hs = np.nanmedian(np.linalg.norm(home - hc[:, None, :], axis=-1), axis=1)
        ass = np.nanmedian(np.linalg.norm(away - ac[:, None, :], axis=-1), axis=1)
        speed = np.concatenate([displacement_speed(home), displacement_speed(away)], axis=1)
        med = np.nanmedian(speed, axis=1)
        p75 = np.nanpercentile(speed, 75, axis=1)
    columns = (
        b_seen, displacement_speed(ball), nh, na, np.fmin(nh, na), na - nh, np.abs(na - nh),
        h_seen.sum(axis=1), a_seen.sum(axis=1), med, p75,
        np.isfinite(speed).sum(axis=1), hs, ass, np.linalg.norm(hc - ac, axis=1),
    )
    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (len(ball), len(FEATURE_NAMES)) or np.isinf(result).any():
        raise ValueError("Bentuk/nilai fitur tidak valid.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", choices=sorted(MATCHES), default="J03WMX")
    parser.add_argument("--folder", type=Path, default=None)
    args = parser.parse_args()
    match_id = "DFL-MAT-" + args.match
    folder = (args.folder or Path(__file__).resolve().parent / "data/external/idsse" / args.match).resolve()
    expected_files = MATCHES[args.match]
    hashes = {name: hash_file(folder / name) for name in expected_files}
    for name, expected in expected_files.items():
        if hashes[name]["md5"] != expected:
            raise ValueError("Hash tidak cocok: " + name)
    root = ET.parse(folder / "match_information.xml").getroot()
    general = root.find(".//General")
    if general is None or general.get("MatchId") != match_id:
        raise ValueError("Metadata bukan " + args.match)
    home_id = general.get("HomeTeamId")
    away_id = general.get("GuestTeamId") or general.get("AwayTeamId")
    if not home_id or not away_id or home_id == away_id:
        raise ValueError("Pemetaan tim tidak valid.")
    print("Pertandingan:", general.get("MatchTitle"), flush=True)
    data = read_coordinates_and_targets(folder / "positions.xml", home_id, away_id, match_id)
    matrices, labels, frame_ids, halves = [], [], [], []
    summaries = []
    for half, period in enumerate(("firstHalf", "secondHalf"), 1):
        d = data.pop(period)
        features = make_features(d["teams"][home_id], d["teams"][away_id], d["ball"])
        matrices.append(features)
        labels.append(d["target"])
        frame_ids.append(d["frame"])
        halves.append(np.full(len(features), half, dtype=np.int8))
        counts = dict(Counter(d["target"].tolist()))
        summaries.append({"half": half, "rows": len(features), "targets": counts})
        print("Babak {}: {} baris | target {}".format(half, len(features), counts), flush=True)
        del d
    X, y = np.concatenate(matrices), np.concatenate(labels)
    frame, half = np.concatenate(frame_ids), np.concatenate(halves)
    out = folder / "features_v1"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", X=X, feature_names=np.array(FEATURE_NAMES),
                        frame=frame, half=half, match_id=np.array(match_id))
    np.savez_compressed(out / "targets.npz", y=y, frame=frame, half=half,
                        match_id=np.array(match_id))
    report = {
        "match_id": match_id, "source_doi": "10.6084/m9.figshare.28196177.v1",
        "credit": "Deutsche Fussball Liga; Bassek, Rein, Weber & Memmert (2025)",
        "license": "CC BY 4.0", "input_hashes": hashes, "numpy": np.__version__,
        "script_sha256": hash_file(Path(__file__))["sha256"],
        "rows": len(X), "feature_names": FEATURE_NAMES, "player_only_features": PLAYER_ONLY_NAMES,
        "periods": summaries,
        "missing_values_by_feature": dict(zip(FEATURE_NAMES, np.isnan(X).sum(axis=0).tolist())),
        "feature_definition": "X/Y coordinates only. Speeds are 1-second endpoint displacement / 1 second; no cross-half pairs. No XML S, Z, M, BallStatus, BallPossession, IDs, timestamps or half/frame numbers used as predictor columns.",
        "target_policy": "BallStatus=0 -> D; status=1 with BallPossession=1/2 -> H/A; otherwise X; never filled.",
        "training_performed": False, "challenge_data_read": False,
        "frame_clock_checked_hz": FPS,
        "warning": "One external match only; not yet an independent train/validation split or evidence of challenge accuracy.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFITUR SELESAI: {} baris x {} fitur".format(*X.shape))
    print("Label:", dict(Counter(y.tolist())))
    print("Frame bola teramati:", int(X[:, 0].sum()))
    print("Fitur gerak tanpa sampel (awal babak):", int((X[:, 11] == 0).sum()))
    print("Fitur:", out / "features.npz")
    print("Target TERPISAH:", out / "targets.npz")
    print("Laporan:", out / "report.json")
    print("Belum training. Tidak membaca data/GT pertandingan challenge.")


if __name__ == "__main__":
    main()
