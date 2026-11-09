#!/usr/bin/env python3
"""
eval_thresholds.py — measure how well ADTOF thresholds reproduce YOUR finished
charts, and learn better ones from them.

Point it at song folders you've already corrected in Moonscraper:

    python eval_thresholds.py "out/Ride - Leave Them All Behind" "out/Hum - Dreamboat" ...

Each folder needs notes.chart (your corrected version) plus either
_work/activations.npy (saved by chart_drums.py) or audio to re-run ADTOF on
(_work/padded.wav, else song.ogg — both share the chart's padded timeline).

Objective: EDITS = extra notes + missed notes, i.e. your Moonscraper cleanup
work (a mislabelled hit costs 2: delete + add). Edits are additive across
classes and songs, so every column below is directly comparable and 'oracle'
is a true floor. F-measure is shown too, for familiarity.

For every drum class it reports edits per 100 of your notes, and F, for:
  default   ADTOF's shipped thresholds
  auto      per-song --auto-thresholds (unsupervised)
  global    one threshold per class learned from all your songs (in-sample)
  LOO       same, but each song scored with thresholds learned from the OTHER
            songs — the honest estimate of what 'global' does on a new song
  oracle    best possible threshold per song per class — the FLOOR on edits for
            any threshold-only method. If oracle is far below your target, the
            remaining errors are misclassifications/misses thresholds can't fix.

Also reports the systematic timing lag between ADTOF onsets and your chart
(-> a measured --note-shift-ms) and the most common class confusions.

Caveat: your corrected charts started life as default-threshold output, so
borderline notes you never noticed count as 'correct'. That biases the score
slightly toward the defaults — any improvement found here is, if anything,
understated. The learned thresholds also encode your taste (e.g. dropping ghost
notes), which no unsupervised method can know.
"""

import argparse
import os
import sys

import numpy as np

import chart_io
import chart_drums as cd

GRID = np.round(np.arange(0.05, 0.801, 0.01), 2)
MIN_GT_TO_LEARN = 20          # classes with fewer notes keep the default


def load_song(folder, device):
    chart = os.path.join(folder, "notes.chart")
    secs = chart_io.parse_sections(chart_io.read_text(chart))
    props = chart_io.song_props(secs)
    res = int(float(props.get("Resolution", 192)))
    # Positive Offset delays notes relative to audio (Clone Hero convention);
    # the lag estimate below absorbs any residual constant error either way.
    offset = float(props.get("Offset", 0) or 0)
    tm = chart_io.TempoMap(chart_io.sync_tempos(secs), res)
    hits = chart_io.drum_hits(secs)

    work = os.path.join(folder, "_work")
    cache = os.path.join(work, "activations.npy")
    if os.path.exists(cache):
        act = np.load(cache)
    else:
        audio = next((p for p in (os.path.join(work, "padded.wav"),
                                  os.path.join(folder, props.get("MusicStream") or "song.ogg"))
                      if os.path.exists(p)), None)
        if audio is None:
            sys.exit(f"{folder}: no activations.npy and no audio to run ADTOF on")
        os.makedirs(work, exist_ok=True)
        print(f"  running ADTOF on {os.path.basename(audio)} ...")
        act = cd.compute_activations(audio, work, device)
    return hits, tm, offset, act


def count_tp(det, gt, tol):
    """One-to-one in-order matching of two sorted onset arrays within tol."""
    i = j = tp = 0
    while i < len(det) and j < len(gt):
        d = det[i] - gt[j]
        if abs(d) <= tol:
            tp += 1; i += 1; j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return tp


def match_masks(det, gt, tol):
    """Same matching as count_tp, returning (det_matched, gt_matched) masks."""
    dm, gm = np.zeros(len(det), bool), np.zeros(len(gt), bool)
    i = j = 0
    while i < len(det) and j < len(gt):
        d = det[i] - gt[j]
        if abs(d) <= tol:
            dm[i] = gm[j] = True; i += 1; j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return dm, gm


def best_plateau(edits):
    """Index minimising edits; middle of the optimal plateau (robust to ties)."""
    e = np.asarray(edits)
    idx = np.flatnonzero(e == e.min())
    return int(idx[len(idx) // 2])


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def estimate_lag(det_by_cls, gt_by_cls, window=0.15):
    diffs = []
    for c, gt in gt_by_cls.items():
        det = det_by_cls.get(c, np.array([]))
        if det.size == 0:
            continue
        for g in gt:
            k = np.searchsorted(det, g)
            near = [det[m] - g for m in (k - 1, k) if 0 <= m < det.size]
            if near:
                d = min(near, key=abs)
                if abs(d) <= window:
                    diffs.append(d)
    return (float(np.median(diffs)) if diffs else 0.0), len(diffs)


def main():
    ap = argparse.ArgumentParser(description="Evaluate/learn ADTOF thresholds from corrected charts")
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--tol", type=float, default=50.0, help="match tolerance, ms")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tol = a.tol / 1000.0

    names, defaults = cd.adtof_labels()
    songs = []
    for folder in a.folders:
        print(f"loading {folder}")
        hits, tm, offset, act = load_song(folder, a.device)
        gt = {c: np.array(sorted(tm.tick_to_sec(t) + offset for t, k in hits if k == c))
              for c in names}
        preps = {c: cd.peak_prep(act[:, i]) for i, c in enumerate(names)}
        base = {c: cd.pick_frames(preps[c], defaults[c]) / cd.FPS for c in names}
        lag, n = estimate_lag(base, gt)
        songs.append(dict(name=os.path.basename(os.path.normpath(folder)),
                          gt=gt, preps=preps, lag=lag, nlag=n))

    # ---- sweep: counts[song][class] = (tp[], fp[], fn[]) over GRID ---------- #
    for s in songs:
        s["counts"] = {}
        for c in names:
            gt, tps, fps, fns = s["gt"][c], [], [], []
            for thr in GRID:
                det = cd.pick_frames(s["preps"][c], thr) / cd.FPS - s["lag"]
                tp = count_tp(det, gt, tol)
                tps.append(tp); fps.append(det.size - tp); fns.append(gt.size - tp)
            s["counts"][c] = tuple(map(np.array, (tps, fps, fns)))

    def at(s, c, thr):
        det = cd.pick_frames(s["preps"][c], thr) / cd.FPS - s["lag"]
        tp = count_tp(det, s["gt"][c], tol)
        return tp, det.size - tp, s["gt"][c].size - tp

    def best_idx(song_list, c):
        fp = sum(s["counts"][c][1] for s in song_list)
        fn = sum(s["counts"][c][2] for s in song_list)
        return best_plateau(fp + fn)

    n_gt = {c: sum(s["gt"][c].size for s in songs) for c in names}
    learned = {c: (float(GRID[best_idx(songs, c)]) if n_gt[c] >= MIN_GT_TO_LEARN
                   else defaults[c]) for c in names}

    strategies = ["default", "auto", "global", "LOO", "oracle"]
    tot = {k: {c: np.zeros(3) for c in names} for k in strategies}
    per_song = {k: [] for k in strategies}
    for s in songs:
        others = [o for o in songs if o is not s]
        song_tot = {k: np.zeros(3) for k in strategies}
        for c in names:
            auto_thr, _ = cd.auto_threshold(s["preps"][c], defaults[c])
            loo_thr = (float(GRID[best_idx(others, c)])
                       if others and sum(o["gt"][c].size for o in others) >= MIN_GT_TO_LEARN
                       else defaults[c])
            tp, fp, fn = s["counts"][c]
            oi = best_plateau(fp + fn)
            res = {"default": at(s, c, defaults[c]), "auto": at(s, c, auto_thr),
                   "global": at(s, c, learned[c]), "LOO": at(s, c, loo_thr),
                   "oracle": (tp[oi], fp[oi], fn[oi])}
            for k, v in res.items():
                tot[k][c] += v
                song_tot[k] += v
        g = sum(s["gt"][c].size for c in names)
        for k in strategies:
            per_song[k].append(100.0 * (song_tot[k][1] + song_tot[k][2]) / max(1, g))

    # ---- report ------------------------------------------------------------ #
    print(f"\nTiming: ADTOF onsets vs your charts (positive = ADTOF late)")
    for s in songs:
        print(f"  {s['name'][:40]:40s} {s['lag'] * 1000:+6.1f} ms  ({s['nlag']} matched notes)")
    med = float(np.median([s["lag"] for s in songs]))
    print(f"  -> suggested --note-shift-ms {-med * 1000:+.0f}")

    def ed(v, g):
        return 100.0 * (v[1] + v[2]) / max(1, g)

    print(f"\nEdits per 100 of your notes (lower is better; tolerance {a.tol:.0f} ms)")
    print(f"  {'class':7s} {'notes':>6s} " + " ".join(f"{k:>8s}" for k in strategies)
          + "   learned thr")
    for c in names:
        row = " ".join(f"{ed(tot[k][c], n_gt[c]):8.1f}" for k in strategies)
        tag = "" if n_gt[c] >= MIN_GT_TO_LEARN else "  (too few notes; default kept)"
        print(f"  {c:7s} {n_gt[c]:6d} {row}   {defaults[c]:.2f} -> {learned[c]:.2f}{tag}")
    G = sum(n_gt.values())
    allv = {k: sum(tot[k].values()) for k in strategies}
    print(f"  {'ALL':7s} {G:6d} " + " ".join(f"{ed(allv[k], G):8.1f}" for k in strategies))
    print(f"  {'ALL F':7s} {'':6s} " + " ".join(f"{f1(*allv[k]):8.3f}" for k in strategies))

    if len(songs) > 1:
        print("\nPer-song edits per 100 notes")
        for i, s in enumerate(songs):
            print(f"  {s['name'][:40]:40s} " + " ".join(f"{per_song[k][i]:8.1f}" for k in strategies))
    else:
        print("\n(LOO needs 2+ songs; with one song it equals default.)")

    # ---- confusions at learned thresholds ---------------------------------- #
    # A miss of class c counts as a confusion with class o only if o has an
    # UNMATCHED (extra) detection at that moment — legit co-occurring notes
    # (hats under a tom fill) don't count.
    conf, fn_total = {}, 0
    for s in songs:
        dets, extra = {}, {}
        for c in names:
            d = cd.pick_frames(s["preps"][c], learned[c]) / cd.FPS - s["lag"]
            dm, gm = match_masks(d, s["gt"][c], tol)
            dets[c], extra[c] = d, d[~dm]
            s.setdefault("_gm", {})[c] = gm
        for c in names:
            for g in s["gt"][c][~s["_gm"][c]]:
                fn_total += 1
                for o in names:
                    if o != c and extra[o].size and np.min(np.abs(extra[o] - g)) <= tol:
                        conf[(c, o)] = conf.get((c, o), 0) + 1
    if conf:
        print(f"\nMisses where ADTOF put an extra note of another class instead "
              f"(of {fn_total} misses at learned thresholds):")
        for (c, o), n in sorted(conf.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  your {c:7s} <- ADTOF said {o:7s}: {n}")

    named = ",".join(f"{c}={learned[c]:.2f}" for c in names)
    print(f"\nLearned thresholds:\n  --adtof-thresholds {named}")


if __name__ == "__main__":
    main()
