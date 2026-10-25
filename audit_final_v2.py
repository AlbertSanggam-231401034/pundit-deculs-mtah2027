"""Audit kandidat v2 dan reproduksi lokal. Tidak membaca GT atau mengubah prediksi."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
N = 199319
FPS = 25
BLOCKS = ((1, 11185, 90652), (2, 110808, 190198))  # exclusive stop
EXPECTED_V2 = "7967963a92f429a0b4c53433e6a418dcd9a34802cfb0958a29acfface96f61af"
LABELS = "HADX"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path, n=N):
    labels = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        if next(rows, None) != ["frame", "label"]:
            raise ValueError("Header bukan frame,label: " + str(path))
        for i, row in enumerate(rows):
            if len(row) != 2 or row[0] != str(i) or row[1] not in LABELS:
                raise ValueError("Frame/label tidak valid pada baris " + str(i + 2))
            labels.append(row[1])
    if len(labels) != n:
        raise ValueError("Jumlah baris salah: " + str(len(labels)))
    return labels


def runs(labels):
    if not labels:
        return []
    out, first = [], 0
    for i in range(1, len(labels)):
        if labels[i] != labels[first]:
            out.append((first, i, labels[first]))
            first = i
    out.append((first, len(labels), labels[first]))
    return out


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def describe(labels, blocks=BLOCKS):
    profiles, segments, transitions, sandwiches = [], [], Counter(), []
    inside = bytearray(len(labels))
    for half, first, stop in blocks:
        inside[first:stop] = b"\1" * (stop - first)
        part = labels[first:stop]
        rs = runs(part)
        counts = Counter(part)
        profiles.append({"half": half, "frames": len(part), "label_counts": dict(counts),
                         "label_pct": {k: round(100 * counts[k] / len(part), 3) for k in LABELS},
                         "home_share_predicted_live_pct": round(100 * counts["H"] / max(1, counts["H"] + counts["A"]), 3)})
        for k in LABELS:
            durations = [(b - a) / FPS for a, b, label in rs if label == k]
            segments.append({"half": half, "label": k, "segments": len(durations),
                             "median_s": statistics.median(durations) if durations else None,
                             "p90_s": quantile(durations, .9), "max_s": max(durations, default=None),
                             "under_0_5s": sum(x < .5 for x in durations),
                             "under_1s": sum(x < 1 for x in durations),
                             "under_2s": sum(x < 2 for x in durations)})
        transitions.update(a[2] + "->" + b[2] for a, b in zip(rs, rs[1:]))
        for middle in LABELS:
            for under in (1, 2):
                eligible = [(b - a) / FPS for (_, _, left), (a, b, mid), (_, _, right)
                            in zip(rs, rs[1:], rs[2:])
                            if mid == middle and left == right and left in "HA" and left != mid
                            and (b - a) / FPS < under]
                sandwiches.append({"half": half, "middle_label": middle, "under_s": under,
                                   "surrounded_by_same_team": len(eligible), "total_s": sum(eligible)})
    inside_x = sum(label == "X" and inside[i] for i, label in enumerate(labels))
    outside_non_x = sum(label != "X" and not inside[i] for i, label in enumerate(labels))
    spells = sum(r["segments"] for r in segments if r["label"] in "HA")
    alive = sum(p["label_counts"].get(k, 0) for p in profiles for k in "HA")
    return {"rows": len(labels), "label_counts": dict(Counter(labels)), "profiles": profiles,
            "inside_match_X": inside_x, "outside_match_non_X": outside_non_x,
            "possession_spells": spells, "mean_possession_s": alive / FPS / spells if spells else 0,
            "segments": segments, "transitions": dict(sorted(transitions.items())),
            "short_interruptions": sandwiches,
            "interpretation": "Describes predictions only. Neither spell counts nor short interruptions prove an error. No ground truth used."}


def compare(old, new, blocks=BLOCKS):
    matrix = Counter()
    for _, first, stop in blocks:
        matrix.update(a + "->" + b for a, b in zip(old[first:stop], new[first:stop]))
    return {"rows_old_to_new": dict(sorted(matrix.items())),
            "changed_frames": sum(v for k, v in matrix.items() if k[0] != k[-1]),
            "meaning": "Agreement/disagreement with v1, not correctness."}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    if args.reproduce and args.data_dir is None:
        parser.error("--reproduce memerlukan --data-dir")
    candidate = ROOT / "evaluation/candidates/idsse_v2"
    experiment = ROOT / "evaluation/experiments/idsse_v2"
    old_csv = ROOT / "evaluation/candidates/idsse_markov_v1/Pundit_Deculs.csv"
    csv_path = candidate / "Pundit_Deculs.csv"
    if digest(csv_path) != EXPECTED_V2:
        raise ValueError("SHA256 kandidat berbeda dari output v2 yang telah ditinjau. Tidak ada file yang diubah.")
    saved = read_json(candidate / "report.json")
    frozen = read_json(experiment / "frozen.json")
    if saved["candidate_sha256"] != EXPECTED_V2:
        raise ValueError("Laporan kandidat tidak cocok dengan CSV.")
    if digest(experiment / "model.joblib") != frozen["model_sha256"] or saved["model_sha256"] != frozen["model_sha256"]:
        raise ValueError("Model beku tidak cocok dengan laporan.")
    if digest(ROOT / "pundit_v2.py") != saved["source_sha256"]:
        raise ValueError("Kode pundit_v2.py berubah sejak kandidat dibuat.")
    if saved["signature"] != frozen["signature"]:
        raise ValueError("Tanda tangan fitur tidak cocok.")
    test = read_json(experiment / "test_report.json")
    long = read_json(experiment / "long_gap_report.json")
    if test["frozen"]["model_sha256"] != frozen["model_sha256"] or long["protocol"]["model_sha256"] != frozen["model_sha256"]:
        raise ValueError("Laporan uji memakai model berbeda.")
    labels = read_labels(csv_path)
    summary = describe(labels)
    if summary["inside_match_X"] or summary["outside_match_non_X"]:
        raise ValueError("Label di dalam/luar babak tidak sesuai.")
    audit_dir = ROOT / "evaluation/experiments/idsse_v2_final_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "candidate_sha256": EXPECTED_V2,
              "model_sha256": frozen["model_sha256"], "source_sha256": saved["source_sha256"],
              "audit_script_sha256": digest(__file__), "v2": summary,
              "selected": frozen["selected"], "challenge_gt_read_by_audit": False,
              "external_tests_mean": test["mean_by_scenario"], "long_gap_mean": long["mean_by_scenario"],
              "versions": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
              "reproduction": {"status": "not_requested"}}
    if old_csv.exists():
        old = read_labels(old_csv)
        report["v1"] = describe(old)
        report["comparison"] = compare(old, labels)
    print("KONSISTENSI HASH, MODEL DAN LAPORAN: OK", flush=True)
    print("DISTRIBUSI PER BABAK - BUKAN AKURASI", flush=True)
    for p in summary["profiles"]:
        print(p, flush=True)
    print("SEGMEN PREDIKSI - BUKAN JUMLAH KEJADIAN SEBENARNYA", flush=True)
    for s in summary["segments"]:
        if s["segments"]:
            print(s, flush=True)
    print("TRANSISI:", summary["transitions"], flush=True)
    print("JEDA PENDEK DI ANTARA TIM YANG SAMA:", flush=True)
    for s in summary["short_interruptions"]:
        if s["surrounded_by_same_team"]:
            print(s, flush=True)
    print("Possession v2:", summary["possession_spells"], "| mean_s:", round(summary["mean_possession_s"], 3), flush=True)
    if "v1" in report:
        print("Possession v1:", report["v1"]["possession_spells"], flush=True)
        print("Perbedaan v1 -> v2:", report["comparison"], flush=True)
    write_json(audit_dir / "report.json", report)
    if args.reproduce:
        print("REPRODUKSI: model beku yang sama, proses Python baru, folder output terpisah. Tanpa training.", flush=True)
        with tempfile.TemporaryDirectory(prefix="regenerated_", dir=audit_dir) as work:
            destination = Path(work)
            subprocess.run([sys.executable, str(ROOT / "pundit_v2.py"), "candidate",
                            "--data-dir", str(args.data_dir.resolve()), "--out", str(experiment),
                            "--candidate-out", str(destination)], cwd=ROOT, check=True)
            regenerated = destination / "Pundit_Deculs.csv"
            reproduced_sha = digest(regenerated)
            if reproduced_sha != EXPECTED_V2 or regenerated.read_bytes() != csv_path.read_bytes():
                raise ValueError("REPRODUKSI GAGAL: CSV berbeda. Kandidat asli tetap disimpan.")
            subprocess.run([sys.executable, str(ROOT / "evaluation/validate_submission.py"),
                            str(regenerated)], cwd=ROOT, check=True)
            report["reproduction"] = {"status": "byte_identical", "regenerated_sha256": reproduced_sha,
                                      "official_validator_passed": True,
                                      "scope": "Fresh Python process/output; same local environment, source and frozen model. Not an isolated public repository checkout."}
        if digest(csv_path) != EXPECTED_V2:
            raise ValueError("Kandidat asli berubah selama audit.")
        write_json(audit_dir / "report.json", report)
        print("REPRODUKSI IDENTIK DAN VALIDATOR RESMI: OK", flush=True)
    print("Laporan:", audit_dir / "report.json", flush=True)
    print("Audit tidak menilai akurasi atau menyesuaikan model. Langkah berikutnya: paket kode terpisah dan dokumentasi.", flush=True)


if __name__ == "__main__":
    main()
