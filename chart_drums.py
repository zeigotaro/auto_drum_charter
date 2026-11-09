#!/usr/bin/env python3
"""
chart_drums.py — automate the mechanical parts of pro-drum charting for Clone Hero.

Pipeline (all run on the SAME silence-padded audio so everything stays in sync):

    input .wav
      -> pad N seconds of leading silence, encode full-mix song.ogg   (ffmpeg)
      -> [optional] isolate drum stem, encode drums.ogg               (Demucs)
      -> automatic drum transcription on the FULL MIX                 (ADTOF-pytorch)
           per-class thresholds: defaults, your own, or --auto-thresholds
      -> [optional] split ADTOF's merged cymbal class into ride/crash (DrumSep)
      -> beat tracking -> tempo map (handles non-click / wandering tempo)  (madmom | librosa)
      -> map onsets to ticks + Clone Hero pro-drum frets
      -> write notes.chart  ([Song] / [SyncTrack] / [Events] / [ExpertDrums])
      -> [optional] fetch + syllabify lyrics as untimed placeholders   (ch_lyrics.py)

ADTOF class order is kick, snare, TOM, HIHAT, cymbal (from the package's
LABELS_5 = [35, 38, 47, 42, 49]). Prefer NAMED thresholds to avoid order mix-ups:
    --adtof-thresholds hihat=0.35,tom=0.30

What this deliberately does NOT do:
  - No velocity, accents, or ghost notes. Onsets become plain notes; you add
    dynamics by hand in Moonscraper at your discretion.
  - No section markers. [Events] holds only lyrics (if --lyrics); sections are yours.
  - No double-kick (32) inference. All kicks map to the single kick lane.

Important design note: ADTOF is a "drums-in-the-presence-of-melodic-instruments"
(DTM) model — it was trained on FULL mixes, so it is fed the full padded song,
NOT the isolated drum stem. With velocity dropped, the Demucs step is therefore
optional and only exists to produce drums.ogg for stem-muting in-game.

Dependencies:
  - ffmpeg on PATH
  - (optional) demucs           : pip install demucs        (needs torch)
  - ADTOF-pytorch               : https://github.com/xavriley/ADTOF-pytorch (pip install -e .)
  - (optional) --split-cymbals  : pip install audio-separator  (GPU torch strongly advised)
  - (optional) --lyrics         : pip install pyphen cmudict nltk
  - beat tracking, pick one:
        madmom (preferred for expressive/loose timing) : pip install madmom
        librosa (easy fallback)                        : pip install librosa
  - torch (CUDA auto-detected if present; CPU works, just slower)

Usage:
  python chart_drums.py song.wav -o "out/Artist - Title" \
      --title "Title" --artist "Artist" --year 2004 --charter "you" \
      --tempo-mode follow --time-sig 6/8

Meter: --time-sig writes a proper TS event (6/8 -> "TS 6 3") and maps each
tracked beat to its real note value (a dotted quarter = 288 ticks in 6/8), with
the first detected DOWNBEAT on a barline. Trackers often hear 6/8 as 3/4
(following quarter notes); the run prints a diagnosis and the flag to fix it
(--beat-unit q / e). One time signature per song; add mid-song changes in
Moonscraper (the tempo map is beat-based, so that won't disturb sync).

  --tempo-mode follow    : exact per-beat tempo events, deduped at millibpm
                           precision (tight sync to a drummer w/o a click track)
  --tempo-mode constant  : single median BPM (clean map for click-track songs)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

RESOLUTION = 192          # ticks per quarter note (Moonscraper default)
LEADING_SILENCE_S = 5.0   # matches your manual 5s pad; baked into the audio

# ADTOF-pytorch (xavriley) outputs a MIDI file with 5 drum classes written as
# General MIDI percussion pitches. We map by PITCH (unambiguous, standard) rather
# than by output-column order, so the mapping is robust regardless of internal
# channel ordering. Ranges/sets are used so that whichever representative pitch
# the port emits for toms/cymbals still lands in the right bucket.
#
# To verify on your machine: transcribe one song, open the .mid, and check which
# pitches actually appear; adjust the sets below if anything is off. Kick/snare/
# hi-hat (35/36, 38/40, 42/44/46) are standard; toms and cymbals are the ones
# worth a glance since those are ADTOF's weak classes anyway.
GM_PITCH_TO_CLASS = {
    35: "kick",  36: "kick",
    38: "snare", 40: "snare",
    42: "hihat", 44: "hihat", 46: "hihat",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "cymbal", 51: "cymbal", 52: "cymbal", 53: "cymbal", 55: "cymbal",
    57: "cymbal", 59: "cymbal",
}

# Clone Hero pro-drum mapping: class -> list of .chart frets to emit at each hit.
# Frets: 0=kick 1=red 2=yellow 3=blue 4=green.
# Cymbal-toggle markers (so a pad reads as a cymbal, not a tom): 66=yellow 67=blue 68=green.
# Default below: hi-hat = yellow cymbal, tom = blue (tom), cymbal = green cymbal.
# With --split-cymbals, ride = blue cymbal and crash = green cymbal (standard
# pro-drums layout). Toms step aside to green/yellow tom if their lane is a
# cymbal at that tick (see onsets_to_notes).
CLASS_TO_FRETS = {
    "kick":   [0],
    "snare":  [1],          # red, tom-type, no cymbal marker
    "hihat":  [2, 66],      # yellow + yellow-cymbal marker
    "tom":    [3],          # blue tom
    "cymbal": [4, 68],      # green + green-cymbal marker (ride+crash merged)
    "ride":   [3, 67],      # blue + blue-cymbal marker   (--split-cymbals)
    "crash":  [4, 68],      # green + green-cymbal marker (--split-cymbals)
}
TOM_LANES = [(3, 67), (4, 68), (2, 66)]   # (tom fret, cymbal marker that blocks it)

DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"   # audio-separator catalog name


@dataclass
class Meta:
    title: str = "Unknown"
    artist: str = "Unknown"
    album: str = ""
    year: str = ""
    charter: str = ""
    music_stream: str = "song.ogg"


# --------------------------------------------------------------------------- #
# Small shell helper                                                          #
# --------------------------------------------------------------------------- #

def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- #
# Step 1 — pad leading silence, encode full-mix ogg                           #
# --------------------------------------------------------------------------- #

def pad_and_encode(input_wav, outdir, workdir):
    """Returns (padded_wav_path, song_ogg_path). adelay all=true prepends silence
    regardless of channel count, so audio + transcription + beats share one clock."""
    ms = int(LEADING_SILENCE_S * 1000)
    padded_wav = os.path.join(workdir, "padded.wav")
    song_ogg = os.path.join(outdir, "song.ogg")
    run(["ffmpeg", "-y", "-i", input_wav,
         "-af", f"adelay={ms}:all=true", padded_wav])
    run(["ffmpeg", "-y", "-i", padded_wav,
         "-c:a", "libvorbis", "-q:a", "6", song_ogg])
    return padded_wav, song_ogg


# --------------------------------------------------------------------------- #
# Step 2 — optional drum stem -> drums.ogg (NOT fed to ADTOF)                  #
# --------------------------------------------------------------------------- #

def make_drum_stem_ogg(padded_wav, outdir, workdir, device):
    if shutil.which("demucs") is None:
        print("  ! demucs not found; skipping drums.ogg")
        return None, None
    run(["demucs", "--two-stems=drums", "-d", device,
         "-o", workdir, padded_wav])
    name = os.path.splitext(os.path.basename(padded_wav))[0]
    drums_wav = os.path.join(workdir, "htdemucs", name, "drums.wav")
    if not os.path.exists(drums_wav):
        print("  ! expected drum stem missing; skipping drums.ogg")
        return None, None
    drums_ogg = os.path.join(outdir, "drums.ogg")
    run(["ffmpeg", "-y", "-i", drums_wav,
         "-c:a", "libvorbis", "-q:a", "6", drums_ogg])
    return drums_wav, drums_ogg


# --------------------------------------------------------------------------- #
# Step 3 — ADTOF activations, peak picking, thresholds                        #
# --------------------------------------------------------------------------- #

FPS = 100
# Mirrors adtof_pytorch.post_processing.PeakPicker's settings exactly
# (pre_avg 0.1s, post_avg 0.01s, pre_max 0.02s, post_max 0.01s, combine 0.02s).
_PRE_AVG, _POST_AVG, _PRE_MAX, _POST_MAX, _COMBINE = 10, 1, 2, 1, 2   # frames


def adtof_labels():
    """(class names in ADTOF column order, default thresholds dict)."""
    from adtof_pytorch import LABELS_5, FRAME_RNN_THRESHOLDS
    names = [GM_PITCH_TO_CLASS[p] for p in LABELS_5]     # kick snare tom hihat cymbal
    return names, dict(zip(names, FRAME_RNN_THRESHOLDS))


def compute_activations(audio, workdir, device, reuse=False):
    """Raw per-frame class probabilities [T, 5] at 100 fps, cached in _work so
    re-runs with new thresholds (and eval_thresholds.py) skip the model."""
    cache = os.path.join(workdir, "activations.npy")
    if reuse and os.path.exists(cache):
        print("      reusing cached activations")
        return np.load(cache)
    from adtof_pytorch import transcribe_to_midi
    pred = transcribe_to_midi(audio, os.path.join(workdir, "_unused.mid"),
                              return_activations=True, device=device)
    act = np.asarray(pred)[0].astype(np.float32)
    np.save(cache, act)
    return act


def peak_prep(act_1d):
    """Threshold-independent part of ADTOF's peak picker: returns (proc, is_lmax).
    proc = activation minus its local moving average (what thresholds apply to)."""
    x = np.asarray(act_1d, dtype=np.float32).reshape(-1)
    k = np.ones(_PRE_AVG + 1 + _POST_AVG, dtype=np.float32) / (_PRE_AVG + 1 + _POST_AVG)
    avg = np.convolve(np.pad(x, (_PRE_AVG, _POST_AVG), mode="edge"), k, mode="valid")
    proc = np.maximum(0.0, x - avg.astype(np.float32))
    p = np.pad(proc, (_PRE_MAX, _POST_MAX), mode="edge")
    wmax = np.max(np.stack([p[i:i + len(proc)] for i in range(_PRE_MAX + 1 + _POST_MAX)]), axis=0)
    return proc, proc >= wmax


def pick_frames(prep, thr):
    """Peak frames at threshold thr (identical to ADTOF's NotePeakPickingProcessor)."""
    proc, lmax = prep
    idx = np.where(lmax & (proc >= thr))[0]
    if idx.size == 0:
        return idx
    kept, group = [], [int(idx[0])]
    for i in idx[1:]:
        if i - group[-1] <= _COMBINE:
            group.append(int(i))
        else:
            kept.append(max(group, key=lambda j: proc[j]))
            group = [int(i)]
    kept.append(max(group, key=lambda j: proc[j]))
    return np.asarray(kept)


def resolve_thresholds(spec, defaults, names):
    """spec: None | 'a,b,c,d,e' (ADTOF order) | 'hihat=0.35,tom=0.3' (named)."""
    thr = dict(defaults)
    if not spec:
        return thr
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if all("=" in p for p in parts):
        for p in parts:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            if k not in thr:
                sys.exit(f"Unknown threshold class '{k}' (valid: {', '.join(names)})")
            thr[k] = float(v)
    elif len(parts) == len(names) and not any("=" in p for p in parts):
        thr.update(zip(names, map(float, parts)))
    else:
        sys.exit(f"--adtof-thresholds: give {len(names)} values in order "
                 f"{','.join(names)} or named pairs like hihat=0.35")
    return thr


def auto_threshold(prep, base, floor=0.05, min_cands=40, min_eta=0.5,
                   lo=1.0, hi=2.0):
    """Per-song threshold from the song's own peak-height distribution.

    Takes every local peak above a low floor, and runs Otsu's method on log peak
    height to find the natural split between 'noise bumps' and 'real hits'.
    Guard rails: falls back to `base` if there are too few peaks or the
    distribution isn't clearly two-humped (Otsu separability eta < min_eta),
    and clamps the result to [lo*base, hi*base] so one odd song can't go wild.
    lo=1.0 means auto only ever RAISES thresholds: Otsu splits noise from
    everything real, ghost notes included, so on strong classes it tends to
    lower the cut and invite ghost-note clutter. Extra notes are the failure
    being fixed here; lowering is left to thresholds learned from your charts.
    Returns (threshold, reason).

    Unsupervised: it finds separable clusters, not YOUR charting taste —
    eval_thresholds.py measures whether it actually beats fixed thresholds.
    """
    proc, lmax = prep
    h = proc[lmax & (proc >= floor)]
    if h.size < min_cands:
        return base, f"kept base ({h.size} peaks)"
    x = np.log(h)
    hist, edges = np.histogram(x, bins=128)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m = np.cumsum(p * centers)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    sb = np.nan_to_num(sb)
    k = int(np.argmax(sb))
    var_t = float(np.sum(p * (centers - mt) ** 2))
    eta = float(sb[k] / var_t) if var_t > 0 else 0.0
    if eta < min_eta:
        return base, f"kept base (eta {eta:.2f})"
    t = float(np.exp(edges[k + 1]))
    clamped = min(max(t, lo * base), hi * base)
    note = f"otsu eta {eta:.2f}" + ("" if clamped == t else f", clamped from {t:.2f}")
    return round(clamped, 3), note


def transcribe(act, names, thresholds):
    """-> sorted [(time_s, class)] using the given per-class thresholds."""
    onsets = []
    for c, name in enumerate(names):
        for f in pick_frames(peak_prep(act[:, c]), thresholds[name]):
            onsets.append((f / FPS, name))
    onsets.sort(key=lambda x: x[0])
    return onsets


HAND_CLASSES = ("snare", "tom", "hihat", "cymbal")


def onset_strengths(onsets, act, names, thresholds):
    """Peak height of each onset relative to its class threshold (>= 1), so
    classes with different thresholds compare fairly. Parallel to onsets."""
    preps = {n: peak_prep(act[:, c])[0] for c, n in enumerate(names)}
    out = []
    for t, name in onsets:
        f = min(len(preps[name]) - 1, int(round(t * FPS)))
        out.append(float(preps[name][f]) / max(thresholds[name], 1e-6))
    return out


def limit_hands(onsets, act, names, thresholds, window_s=0.025, max_hands=2):
    """A drummer has two hands: where more than `max_hands` different hand
    classes (everything but kick) fire together, keep the strongest and drop
    the rest. 'Together' = within window_s, since ADTOF's classes often peak a
    frame apart for what is one stroke. Returns (onsets, n_dropped)."""
    strengths = onset_strengths(onsets, act, names, thresholds)
    drop, i = set(), 0
    hands = [(j, t, n) for j, (t, n) in enumerate(onsets) if n in HAND_CLASSES]
    while i < len(hands):
        k = i + 1
        while k < len(hands) and hands[k][1] - hands[i][1] <= window_s:
            k += 1
        best = {}                                   # class -> strongest strength
        for j, t, n in hands[i:k]:
            best[n] = max(best.get(n, 0.0), strengths[j])
        if len(best) > max_hands:
            losers = set(sorted(best, key=best.get, reverse=True)[max_hands:])
            drop.update(j for j, _, n in hands[i:k] if n in losers)
        i = k
    return [o for j, o in enumerate(onsets) if j not in drop], len(drop)


# --------------------------------------------------------------------------- #
# Step 3b — optional ride/crash split via DrumSep                             #
# --------------------------------------------------------------------------- #

def drumsep_stems(drums_wav, workdir):
    """Run the 6-stem DrumSep model on the isolated drum stem; {stem: wav_path}."""
    outdir = os.path.join(workdir, "drumsep")
    os.makedirs(outdir, exist_ok=True)

    def found():
        stems = {}
        for f in os.listdir(outdir):
            low = f.lower()
            if not low.endswith(".wav"):
                continue
            m = re.search(r"\(([^)]+)\)", low)
            label = m.group(1).strip() if m else low
            for key in ("kick", "snare", "toms", "hh", "ride", "crash"):
                if label == key or (not m and key in label):
                    stems[key] = os.path.join(outdir, f)
        return stems

    stems = found()
    if not ("ride" in stems and "crash" in stems):
        if shutil.which("audio-separator") is None:
            sys.exit("--split-cymbals needs: pip install audio-separator")
        run(["audio-separator", drums_wav, "--model_filename", DRUMSEP_MODEL,
             "--output_dir", outdir, "--output_format", "WAV"])
        stems = found()
    if not ("ride" in stems and "crash" in stems):
        sys.exit(f"DrumSep output not recognised in {outdir}; found {sorted(stems)}")
    return stems


def _load_mono(path, sr):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _envelope(y, hop):
    n = len(y) // hop
    rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return np.maximum(0.0, np.diff(rms, prepend=rms[:1]))     # onset-ish flux


def stem_lag_seconds(ref, est, sr, hop=220, max_lag_s=0.3):
    """Seconds by which `est` is LATE relative to `ref` (negative = early).
    Guards against separators that trim/shift output (reported for DrumSep)."""
    a, b = _envelope(ref, hop), _envelope(est, hop)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    L = int(max_lag_s * sr / hop)
    scores = [np.dot(a[:n - k], b[k:]) if k >= 0 else np.dot(a[-k:], b[:n + k])
              for k in range(-L, L + 1)]
    return (int(np.argmax(scores)) - L) * hop / sr


def split_cymbals(onsets, drums_wav, workdir, ride_bias=1.0):
    """Relabel ADTOF 'cymbal' onsets as 'ride' or 'crash' by comparing the two
    DrumSep stems' energy just after each onset. ride_bias > 1 favours ride."""
    sr = 22050
    stems = drumsep_stems(drums_wav, workdir)
    ride, crash = _load_mono(stems["ride"], sr), _load_mono(stems["crash"], sr)
    ref = _load_mono(drums_wav, sr)
    total = np.zeros(max(len(ride), len(crash)), dtype=np.float32)
    for k, pth in stems.items():
        y = ride if k == "ride" else crash if k == "crash" else _load_mono(pth, sr)
        if len(y) > len(total):
            total = np.pad(total, (0, len(y) - len(total)))
        total[:len(y)] += y
    lag = stem_lag_seconds(ref, total, sr)
    if abs(lag) > 0.005:
        print(f"      DrumSep stems offset {lag * 1000:+.0f} ms vs drum stem; compensating")

    def energy(y, t):
        a, b = int((t + lag - 0.01) * sr), int((t + lag + 0.10) * sr)
        seg = y[max(0, a):max(0, b)]
        return float(np.sum(seg * seg)) if seg.size else 0.0

    out, n_ride, n_crash = [], 0, 0
    for t, name in onsets:
        if name == "cymbal":
            er, ec = energy(ride, t), energy(crash, t)
            name = "ride" if er * ride_bias > ec else "crash"
            n_ride += name == "ride"
            n_crash += name == "crash"
        out.append((t, name))
    print(f"      cymbals -> {n_ride} ride / {n_crash} crash")
    return out


# --------------------------------------------------------------------------- #
# Time signatures                                                             #
# --------------------------------------------------------------------------- #

# Note value of one TRACKED beat, in quarter notes. Beat trackers follow the
# felt pulse: a quarter in 4/4, but a DOTTED quarter in 6/8, 9/8, 12/8, 3/8.
BEAT_UNITS = {"h": 2.0, "q": 1.0, "dq": 1.5, "e": 0.5, "de": 0.75, "s": 0.25}
BEAT_UNIT_NAMES = {2.0: "half", 1.0: "quarter", 1.5: "dotted-quarter",
                   0.5: "eighth", 0.75: "dotted-eighth", 0.25: "sixteenth"}


@dataclass
class TimeSig:
    num: int
    den: int
    beat_quarters: float           # tracked-beat length in quarter notes

    @property
    def measure_ticks(self):       # .chart ticks are always per QUARTER note
        return int(round(self.num * 4 / self.den * RESOLUTION))

    @property
    def beat_ticks(self):
        return int(round(self.beat_quarters * RESOLUTION))

    @property
    def beats_per_bar(self):
        return self.measure_ticks // self.beat_ticks

    @property
    def ts_event(self):            # TS <numerator> <log2 denominator>; x/4 may omit it
        exp = self.den.bit_length() - 1
        return f"TS {self.num}" if self.den == 4 else f"TS {self.num} {exp}"

    def describe(self):
        unit = BEAT_UNIT_NAMES.get(self.beat_quarters, f"{self.beat_quarters}q")
        return (f"{self.num}/{self.den}: tracking {unit} beats, "
                f"{self.beats_per_bar} per bar")


def parse_time_sig(spec, beat_unit=None):
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec or "")
    if not m:
        sys.exit(f"--time-sig must look like 6/8, got {spec!r}")
    num, den = int(m.group(1)), int(m.group(2))
    if num < 1 or den < 1 or den & (den - 1):
        sys.exit(f"--time-sig {spec}: denominator must be a power of 2")
    if beat_unit:
        bq = BEAT_UNITS[beat_unit]
    elif den >= 8 and num % 3 == 0:
        bq = 3 * 4 / den           # compound: 3/8, 6/8, 9/8, 12/8 -> dotted quarter
    else:
        bq = 4 / den               # simple: the denominator note (q in x/4, e in 7/8)
    ts = TimeSig(num, den, bq)
    if abs(bq * RESOLUTION - ts.beat_ticks) > 1e-6 or ts.measure_ticks % ts.beat_ticks:
        hint = "e" if den == 8 else "s" if den == 16 else "q"
        sys.exit(f"{num}/{den} isn't a whole number of "
                 f"{BEAT_UNIT_NAMES.get(bq, bq)} beats; try --beat-unit {hint}")
    return ts


def hits_per_pulse(onsets, beats, tol=0.15, need=0.85):
    """Coarsest grid the hits sit on, as divisions per tracked beat: the
    smallest n such that nearly all gaps between hits are whole multiples of
    (beat period / n). 1 -> hits fall only on the pulse, 2 -> halves, etc.
    Robust to sparse parts (gaps of 1, 2 and 3 grid units all fit n)."""
    ts = sorted(t for t, _ in onsets)
    ev = [t for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > 0.03]
    P = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
    ioi = np.diff(ev)
    ioi = ioi[(ioi > 0.06) & (ioi < 2.5 * P)] if P else ioi[:0]
    if ioi.size < 30:
        return None
    for n in (1, 2, 3, 4, 6, 8):
        m = ioi / (P / n)
        if np.mean(np.abs(m - np.round(m)) < tol) >= need:
            return n
    return None


def backbeat_between_beats(onsets, beats):
    """True if snares mostly land halfway BETWEEN tracked beats: the tracker
    is at half tempo, or a 6/8 song is being heard as 3/4."""
    b = np.asarray(beats)
    ph = []
    for t, name in onsets:
        if name != "snare":
            continue
        i = int(np.searchsorted(b, t)) - 1
        if 0 <= i < len(b) - 1:
            ph.append((t - b[i]) / (b[i + 1] - b[i]))
    ph = np.asarray(ph)
    if ph.size < 20:
        return False
    mid = np.mean(np.abs(ph - 0.5) < 0.08)
    on = np.mean((ph < 0.08) | (ph > 0.92))
    return bool(mid > 0.4 and mid > 2 * on)


def subdivision_feel(onsets, beats, tol=0.04):
    """Where do off-beat onsets fall inside each tracked beat?
    Returns ('threes'|'twos'|'unclear', triple_density, duple_density, n).
    Densities are normalised by window width, so 1.0 = chance."""
    b = np.asarray(beats)
    ph = []
    for t, _ in onsets:
        i = int(np.searchsorted(b, t)) - 1
        if 0 <= i < len(b) - 1:
            f = (t - b[i]) / (b[i + 1] - b[i])
            if 0.1 < f < 0.9:
                ph.append(f)
    ph = np.asarray(ph)
    if ph.size < 30:
        return "unclear", 0.0, 0.0, int(ph.size)
    hit = lambda c: np.mean(np.abs(ph - c) < tol)
    width = 0.8                                    # phases kept: (0.1, 0.9)
    tri = (hit(1 / 3) + hit(2 / 3)) / (2 * 2 * tol / width)
    dup = (hit(0.25) + hit(0.5) + hit(0.75)) / (3 * 2 * tol / width)
    feel = "threes" if tri > 1.5 * dup else "twos" if dup > 1.5 * tri else "unclear"
    return feel, float(tri), float(dup), int(ph.size)


# --------------------------------------------------------------------------- #
# Step 4 — beat tracking -> tempo grid                                        #
# --------------------------------------------------------------------------- #

def _madmom_compat_shim():
    """madmom 0.16.1 (2018) predates Python 3.10 and numpy 1.24, which removed
    aliases it still relies on. Restore them at runtime so the already-compiled
    package imports and runs without a rebuild or a Python/numpy downgrade.
    Only invoked on the madmom path, so the librosa path is untouched."""
    import warnings
    import collections, collections.abc
    for _name in ("MutableSequence", "MutableMapping", "Mapping", "Sequence",
                  "Iterable", "Callable", "MutableSet", "Set", "Hashable",
                  "Container", "Sized"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))
    import numpy as np
    # Probing np.float etc. can itself emit the deprecation warning on numpy
    # versions where the aliases still exist; silence it. Where they're truly
    # gone, this restores them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name, _t in (("float", float), ("int", int), ("complex", complex),
                          ("bool", bool), ("object", object), ("str", str)):
            if not hasattr(np, _name):
                setattr(np, _name, _t)


def track_beats(audio_wav, beats_per_bar, backend, min_bpm=None, max_bpm=None,
                transition_lambda=None):
    """Return (beat_times, positions). positions[i] is the beat's 1-based place
    in its bar (madmom) or None (librosa has no downbeats). madmom handles
    expressive, non-click timing noticeably better; librosa is the easy fallback.
    min/max_bpm are in TRACKED-beat units (dotted quarters in 6/8).

    The madmom DBN knobs matter for drifting drummers:
      min_bpm/max_bpm     : constrain the tempo search range (helps avoid
                            half/double-time flips that derail tracking)
      transition_lambda   : tempo-change stiffness. madmom's default (100)
                            prefers steady tempo; LOWER it (e.g. 20-40) to let
                            the tracker bend with a drummer who drifts.
    """
    if backend == "madmom":
        _madmom_compat_shim()  # must run BEFORE importing madmom
        # DBNDownBeatTrackingProcessor needs a 2-column activation (beat +
        # downbeat), which RNNDownBeatProcessor produces. (RNNBeatProcessor is
        # beats-only/1-column and pairs with DBNBeatTrackingProcessor instead.)
        from madmom.features.downbeats import (RNNDownBeatProcessor,
                                               DBNDownBeatTrackingProcessor)
        act = RNNDownBeatProcessor()(audio_wav)   # shape (N, 2)
        # 1 beat per bar (e.g. 3/8 tracked in dotted quarters): every beat is a
        # barline, so bar phase is irrelevant; track in pairs for a stable model.
        kwargs = {"beats_per_bar": [max(2, beats_per_bar)], "fps": 100}
        if min_bpm is not None:
            kwargs["min_bpm"] = min_bpm
        if max_bpm is not None:
            kwargs["max_bpm"] = max_bpm
        if transition_lambda is not None:
            kwargs["transition_lambda"] = transition_lambda
        proc = DBNDownBeatTrackingProcessor(**kwargs)
        downbeats = proc(act)        # rows: [time, beat_position_in_bar]
        times = [float(t) for t, _ in downbeats]
        pos = [int(round(p)) for _, p in downbeats] if beats_per_bar > 1 else None
        return times, pos
    elif backend == "librosa":
        import librosa
        y, sr = librosa.load(audio_wav, sr=None, mono=True)
        _, frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        return [float(t) for t in librosa.frames_to_time(frames, sr=sr)], None
    raise ValueError(backend)


def first_downbeat_index(positions, beats_per_bar, shift=0):
    """Index (0..bpb-1) of the first beat that starts a bar. Uses madmom's
    positions when available; `shift` rotates barlines N beats later (+) or
    earlier (-) for when the tracker's bar phase is wrong, or with librosa."""
    k = 0
    if positions:
        k = next((i for i, p in enumerate(positions[:beats_per_bar]) if p == 1), 0)
    return (k + shift) % beats_per_bar if beats_per_bar > 1 else 0


def build_tempo_map(beats, mode, ts, downbeat_index=0):
    """Build (tempo_events, time_to_tick). tempo_events is a list of (tick, bpm).

    Meter: tracked beat i occupies ts.beat_ticks ticks (288 for a dotted
    quarter at resolution 192), and tempo values are written in QUARTER notes
    per minute, as .chart requires — so 6/8 at 60 dotted-quarter bpm is B 90000.

    Anchoring: the first DOWNBEAT (beats[downbeat_index]) lands exactly on a
    barline, any pickup beats before it fall at the end of the previous bar,
    and a pre-roll tempo at tick 0 makes chart-time(beats[0]) == beats[0]
    exactly, so audio sync is untouched by any of this.

    follow   : beat i -> tick0 + i*beat_ticks; exact per-beat tempo events,
               deduped at the .chart's millibpm precision (cannot drift).
    constant : single median tempo after the pre-roll.
    """
    if len(beats) < 2:
        sys.exit("Too few beats detected to build a tempo map.")
    bt, mt, bq = ts.beat_ticks, ts.measure_ticks, ts.beat_quarters

    intervals = sorted(beats[i + 1] - beats[i] for i in range(len(beats) - 1))
    median_spb = intervals[len(intervals) // 2]

    # --- pre-roll anchor -------------------------------------------------- #
    ref_spb = median_spb if mode == "constant" else (beats[1] - beats[0])
    ideal_first = beats[0] / ref_spb * bt          # where beats[0] would sit at song tempo
    k = downbeat_index
    bars = max(1, round((ideal_first + k * bt) / mt))
    while bars * mt - k * bt <= 0:
        bars += 1
    tick0 = bars * mt - k * bt                     # tick of beats[0]; beats[k] is on a barline
    pre_bpm = 60.0 * (tick0 / RESOLUTION) / beats[0]

    if mode == "constant":
        tempo_events = [(0, pre_bpm), (tick0, 60.0 * bq / median_spb)]

        def time_to_tick(t):
            if t <= beats[0]:
                tick = t / beats[0] * tick0
            else:
                tick = tick0 + (t - beats[0]) / median_spb * bt
            return max(0, int(round(tick)))

    else:  # follow
        tempo_events, last_mbpm = [(0, pre_bpm)], None
        for i in range(len(beats) - 1):
            mbpm = int(round(60000.0 * bq / (beats[i + 1] - beats[i])))
            if mbpm != last_mbpm:
                tempo_events.append((tick0 + i * bt, mbpm / 1000.0))
                last_mbpm = mbpm

        def time_to_tick(t):
            if t <= beats[0]:
                return max(0, int(round(t / beats[0] * tick0)))
            if t >= beats[-1]:
                spb = beats[-1] - beats[-2]
                return int(round(tick0 + (len(beats) - 1) * bt
                                 + (t - beats[-1]) / spb * bt))
            lo, hi = 0, len(beats) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if beats[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            frac = (t - beats[lo]) / (beats[hi] - beats[lo])
            return int(round(tick0 + (lo + frac) * bt))

    return tempo_events, time_to_tick


# --------------------------------------------------------------------------- #
# Step 5 — onsets -> ticked notes                                             #
# --------------------------------------------------------------------------- #

# Candidate grids per quarter note, as (ticks, penalty). A beat uses the grid
# with the lowest mean snap distance + penalty, so straight 16ths win unless
# the hits clearly sit on triplets (64), 32nds (24) or sextuplets (32).
SNAP_GRIDS = [(RESOLUTION // 4, 0.0), (RESOLUTION // 3, 4.0),
              (RESOLUTION // 8, 6.0), (RESOLUTION // 6, 8.0)]


def snap_ticks(ticks, names):
    """Snap raw ticks to a musical grid, choosing the grid beat by beat.
    A grid that would fold two separate strokes of the same drum onto one tick
    (a 32nd-note roll on a 16th grid) is heavily penalised, so fast fills keep
    all their notes. Returns (snapped ticks, {grid: beats that used it})."""
    q, early = RESOLUTION, RESOLUTION // 16        # a hit just before the beat belongs to it
    beats = {}
    for i, tick in enumerate(ticks):
        beats.setdefault((tick + early) // q, []).append(i)
    out, used = list(ticks), {}
    for b, idx in beats.items():
        rel = np.array([ticks[i] - b * q for i in idx], dtype=float)
        best = None
        for g, pen in SNAP_GRIDS:
            s = np.round(rel / g) * g
            seen, clash = set(), 0
            for i, v in zip(idx, s):
                clash += (names[i], v) in seen
                seen.add((names[i], v))
            cost = float(np.mean(np.abs(s - rel))) + pen + 20.0 * clash
            if best is None or cost < best[0]:
                best = (cost, g, s)
        used[best[1]] = used.get(best[1], 0) + 1
        for i, v in zip(idx, best[2]):
            out[i] = int(b * q + v)
    return out, used


def onsets_to_notes(onsets, time_to_tick, snap=True, strengths=None, max_hands=2):
    """Return {tick: set(frets)}. Hits within the same tick merge into a chord.
    With snap, ticks are pulled onto a 16th/triplet/32nd grid first (see
    snap_ticks); if that stacks more than max_hands hand drums on one tick, the
    weakest (by `strengths`, parallel to onsets) are dropped.
    Toms are placed last, on the first lane (blue, green, yellow) whose cymbal
    marker isn't set at that tick — otherwise the marker would turn the tom
    into a cymbal (matters once ride sits on blue with --split-cymbals)."""
    ticks = [time_to_tick(t) for t, _ in onsets]
    names = [n for _, n in onsets]
    if snap and ticks:
        ticks, used = snap_ticks(ticks, names)
        print("      snapped to grid: " + ", ".join(
            f"{n} beats on 1/{4 * RESOLUTION // g}" for g, n in sorted(used.items(), reverse=True)))
    keep = [True] * len(onsets)
    if strengths is not None:
        at = {}
        for i, (tick, n) in enumerate(zip(ticks, names)):
            if n != "kick":
                at.setdefault(tick, {}).setdefault(n, []).append(i)
        n_drop = 0
        for tick, by_class in at.items():
            if len(by_class) > max_hands:
                ranked = sorted(by_class, reverse=True,
                                key=lambda n: max(strengths[i] for i in by_class[n]))
                for n in ranked[max_hands:]:
                    for i in by_class[n]:
                        keep[i] = False
                    n_drop += 1
        if n_drop:
            print(f"      dropped {n_drop} more weakest hits that snapped onto a 3-hand chord")
    notes, toms = {}, set()
    for tick, name, ok in zip(ticks, names, keep):
        if not ok or tick < 0:
            continue
        if name == "tom":
            toms.add(tick)
            continue
        notes.setdefault(tick, set()).update(CLASS_TO_FRETS[name])
    for tick in toms:
        frets = notes.setdefault(tick, set())
        for lane, marker in TOM_LANES:
            if marker not in frets:
                frets.add(lane)
                break
    return notes


# --------------------------------------------------------------------------- #
# Step 6 — write .chart                                                        #
# --------------------------------------------------------------------------- #

def write_chart(path, meta, tempo_events, ts, notes):
    L = []
    L.append("[Song]")
    L.append("{")
    L.append(f'  Name = "{meta.title}"')
    L.append(f'  Artist = "{meta.artist}"')
    L.append(f'  Charter = "{meta.charter}"')
    L.append(f'  Album = "{meta.album}"')
    L.append(f'  Year = ", {meta.year}"' if meta.year else '  Year = ""')
    L.append("  Offset = 0")                       # silence is in the audio; keep 0
    L.append(f"  Resolution = {RESOLUTION}")
    L.append("  Player2 = bass")
    L.append("  Difficulty = 0")
    L.append("  PreviewStart = 0")
    L.append("  PreviewEnd = 0")
    L.append('  Genre = "rock"')
    L.append('  MediaType = "cd"')
    L.append(f'  MusicStream = "{meta.music_stream}"')
    L.append("}")

    L.append("[SyncTrack]")
    L.append("{")
    L.append(f"  0 = {ts.ts_event}")
    for tick, bpm in tempo_events:
        L.append(f"  {tick} = B {int(round(bpm * 1000))}")
    L.append("}")

    # Sections intentionally left for manual editing in Moonscraper.
    L.append("[Events]")
    L.append("{")
    L.append("}")

    L.append("[ExpertDrums]")
    L.append("{")
    for tick in sorted(notes):
        for fret in sorted(notes[tick]):
            L.append(f"  {tick} = N {fret} 0")
    L.append("}")

    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Tags in, song.ini + album art out                                           #
# --------------------------------------------------------------------------- #

def read_tags(path):
    """Metadata embedded in the input file (FLAC/MP3/...), via ffprobe.
    -> ({title, artist, album, year, track, genre}, has_cover_art)."""
    import json
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format_tags:stream_tags:stream_disposition=attached_pic",
             "-of", "json", path],
            check=True, capture_output=True, text=True, encoding="utf-8").stdout
        info = json.loads(out)
    except Exception:
        return {}, False
    streams = info.get("streams", [])
    raw = {}
    for src in [s.get("tags", {}) for s in streams] + [info.get("format", {}).get("tags", {})]:
        raw.update({k.lower(): str(v).strip() for k, v in src.items()})

    def first(*keys):
        return next((raw[k] for k in keys if raw.get(k)), "")

    year = re.search(r"\d{4}", first("originalyear", "originaldate", "date", "year"))
    track = re.match(r"\d+", first("track", "tracknumber"))
    tags = {"title": first("title"), "artist": first("artist", "album_artist"),
            "album": first("album"), "year": year.group(0) if year else "",
            "track": track.group(0) if track else "", "genre": first("genre")}
    cover = any(s.get("disposition", {}).get("attached_pic") for s in streams)
    return tags, cover


def safe_folder_name(name):
    """Strip characters Windows won't allow in a folder name."""
    return re.sub(r'[<>:"/\\|?*]', "", name).rstrip(". ").strip() or "song"


def write_song_ini(outdir, meta, track="", genre="", length_ms=None, difficulty=None):
    """song.ini for Clone Hero's song list. Never overwrites an existing one
    (you may have tuned difficulty / preview time by hand). Returns a status."""
    path = os.path.join(outdir, "song.ini")
    if os.path.exists(path):
        return "song.ini exists, left untouched"
    rows = [("name", meta.title), ("artist", meta.artist), ("album", meta.album),
            ("genre", genre or "Rock"), ("year", meta.year), ("charter", meta.charter),
            ("album_track", track), ("pro_drums", "True"), ("delay", "0"),
            ("preview_start_time", "0"), ("icon", ""), ("loading_phrase", "")]
    if length_ms:
        rows.append(("song_length", str(int(length_ms))))
    if difficulty is not None:
        rows += [("diff_drums", str(difficulty)), ("diff_drums_real", str(difficulty))]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("[song]\n" + "".join(f"{k} = {v}\n" for k, v in rows))
    return "song.ini written" + ("" if difficulty is not None
                                 else " (no difficulty set: add --difficulty 0-6 or edit it)")


def extract_album_art(input_path, outdir):
    """Embedded cover -> album.png (Clone Hero's expected name), if not there yet."""
    if any(os.path.exists(os.path.join(outdir, f"album.{e}")) for e in ("png", "jpg", "jpeg")):
        return False
    dst = os.path.join(outdir, "album.png")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", input_path, "-an",
                        "-frames:v", "1", "-vf", "scale='min(512,iw)':-1", dst],
                       check=True, capture_output=True)
    except Exception:
        return False
    return os.path.exists(dst)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="wav -> Clone Hero pro-drum .chart")
    ap.add_argument("input_wav")
    ap.add_argument("-o", "--outdir", default=None,
                    help="target song folder (default: 'Artist - Title')")
    ap.add_argument("--title", default=None, help="default: the input file's tag")
    ap.add_argument("--artist", default=None, help="default: the input file's tag")
    ap.add_argument("--album", default=None, help="default: the input file's tag")
    ap.add_argument("--year", default=None, help="default: the input file's tag")
    ap.add_argument("--genre", default=None, help="default: the input file's tag")
    ap.add_argument("--charter", default=os.environ.get("CHART_CHARTER", ""),
                    help="default: the CHART_CHARTER environment variable")
    ap.add_argument("--difficulty", type=int, default=None,
                    help="0-6 drum difficulty for song.ini (omitted if not given)")
    ap.add_argument("--tempo-mode", choices=["follow", "constant"], default="follow")
    ap.add_argument("--time-sig", default=None,
                    help="time signature, e.g. 4/4 (default), 3/4, 6/8, 12/8, 7/8")
    ap.add_argument("--beat-unit", choices=sorted(BEAT_UNITS), default=None,
                    help="note value the beat tracker follows (default from --time-sig: "
                         "dq for 3/8 6/8 9/8 12/8, e for 5/8 7/8, q for x/4). Use e if "
                         "the tracker locks onto eighths in a slow 6/8")
    ap.add_argument("--downbeat-shift", type=int, default=0,
                    help="move barlines N tracked beats later (+) or earlier (-); "
                         "needed with librosa, which has no downbeat detection")
    ap.add_argument("--beats-per-bar", type=int, default=None,
                    help="legacy: same as --time-sig N/4")
    ap.add_argument("--beat-backend", choices=["madmom", "librosa"], default="librosa")
    ap.add_argument("--adtof-thresholds", default=None,
                    help="per-class thresholds: named (hihat=0.35,tom=0.3) or 5 values "
                         "in ADTOF order kick,snare,tom,hihat,cymbal")
    ap.add_argument("--auto-thresholds", action="store_true",
                    help="pick per-song thresholds from each class's peak distribution, "
                         "centred on --adtof-thresholds (or ADTOF defaults)")
    ap.add_argument("--reuse-activations", action="store_true",
                    help="reuse _work/activations.npy (fast threshold re-runs)")
    ap.add_argument("--split-cymbals", action="store_true",
                    help="split cymbals into ride (blue) / crash (green) via DrumSep")
    ap.add_argument("--ride-bias", type=float, default=1.0,
                    help="--split-cymbals: >1 favours ride, <1 favours crash")
    ap.add_argument("--lyrics", action="store_true",
                    help="fetch lyrics (LRCLIB), syllabify, add as untimed placeholders")
    ap.add_argument("--lyrics-file", default=None,
                    help="use this lyrics text instead of downloading (implies --lyrics)")
    ap.add_argument("--no-stem-ogg", action="store_true",
                    help="skip Demucs/drums.ogg (full mix is what ADTOF uses anyway)")
    ap.add_argument("--device", default="cuda", help="torch device for adtof/demucs (cuda|cpu)")
    ap.add_argument("--note-shift-ms", type=float, default=0.0,
                    help="shift all transcribed onsets by this many ms before tick "
                         "mapping (negative = earlier; e.g. -50 if charts consistently "
                         "feel late by 50ms)")
    ap.add_argument("--no-snap", action="store_true",
                    help="keep raw note timing instead of snapping to a 16th/triplet/32nd grid")
    ap.add_argument("--no-sections", action="store_true",
                    help="with --lyrics: don't draft section markers from the timed lyrics")
    ap.add_argument("--min-bpm", type=float, default=None,
                    help="madmom only: lower bound of tempo search range (tracked-beat bpm)")
    ap.add_argument("--max-bpm", type=float, default=None,
                    help="madmom only: upper bound of tempo search range (tracked-beat bpm)")
    ap.add_argument("--transition-lambda", type=float, default=None,
                    help="madmom only: tempo-change stiffness (default 100; lower "
                         "e.g. 20-40 to follow a drifting drummer)")
    args = ap.parse_args()
    ts = parse_time_sig(args.time_sig or (f"{args.beats_per_bar}/4" if args.beats_per_bar
                                          else "4/4"), args.beat_unit)

    tags, has_cover = read_tags(args.input_wav)
    for key in ("title", "artist", "album", "year", "genre"):
        if getattr(args, key) is None:
            setattr(args, key, tags.get(key) or ("Unknown" if key in ("title", "artist") else ""))
    if args.outdir is None:
        if "Unknown" in (args.title, args.artist):
            sys.exit("No title/artist tags in the input: pass -o, or --title and --artist")
        args.outdir = safe_folder_name(f"{args.artist} - {args.title}")
    print(f"{args.artist} - {args.title}"
          + (f" ({args.album}, {args.year})" if args.album or args.year else "")
          + f" -> {args.outdir}")

    os.makedirs(args.outdir, exist_ok=True)
    workdir = os.path.join(args.outdir, "_work")
    os.makedirs(workdir, exist_ok=True)

    print("[1/6] padding silence + encoding song.ogg")
    padded_wav, _ = pad_and_encode(args.input_wav, args.outdir, workdir)

    drums_wav = None
    if not args.no_stem_ogg or args.split_cymbals:
        print("[2/6] isolating drum stem -> drums.ogg")
        drums_wav, _ = make_drum_stem_ogg(padded_wav, args.outdir, workdir, args.device)
    else:
        print("[2/6] skipping drum stem")

    print("[3/6] ADTOF transcription on full mix")
    names, defaults = adtof_labels()
    act = compute_activations(padded_wav, workdir, args.device, args.reuse_activations)
    thr = resolve_thresholds(args.adtof_thresholds, defaults, names)
    if args.auto_thresholds:
        for c, name in enumerate(names):
            thr[name], why = auto_threshold(peak_prep(act[:, c]), thr[name])
            print(f"      {name:7s} thr {thr[name]:.3f}  ({why})")
    onsets = transcribe(act, names, thr)
    counts = {n: sum(1 for _, k in onsets if k == n) for n in names}
    print("      " + "  ".join(f"{n}={thr[n]:.2f}:{counts[n]}" for n in names))
    onsets, n_drop = limit_hands(onsets, act, names, thr)
    if n_drop:
        print(f"      dropped {n_drop} weakest hits where 3+ hand drums fired at once")
    strengths = onset_strengths(onsets, act, names, thr)   # stays parallel to onsets below
    if args.split_cymbals:
        if drums_wav is None:
            sys.exit("--split-cymbals needs the Demucs drum stem (is demucs installed?)")
        print("[3b]  splitting cymbals into ride/crash (DrumSep)")
        onsets = split_cymbals(onsets, drums_wav, workdir, args.ride_bias)
    if args.note_shift_ms:
        shift = args.note_shift_ms / 1000.0
        onsets = [(t + shift, name) for t, name in onsets]
        print(f"      shifted all onsets by {args.note_shift_ms:+.0f} ms")
    print(f"      {len(onsets)} onsets")

    print(f"[4/6] beat tracking ({args.beat_backend}) -> tempo map ({args.tempo_mode})")
    print(f"      {ts.describe()}")
    beats, positions = track_beats(padded_wav, ts.beats_per_bar, args.beat_backend,
                                   args.min_bpm, args.max_bpm, args.transition_lambda)
    k = first_downbeat_index(positions, ts.beats_per_bar, args.downbeat_shift)
    if ts.beats_per_bar > 1:
        src_ = ("madmom" if positions else "assumed") + (
            f", shifted {args.downbeat_shift:+d}" if args.downbeat_shift else "")
        print(f"      first downbeat at {beats[k]:.2f}s (beat {k + 1} of {len(beats)}; {src_})")
    tempo_events, time_to_tick = build_tempo_map(beats, args.tempo_mode, ts, k)
    print(f"      {len(beats)} beats, {len(tempo_events)} tempo events")

    feel, tri, dup, n = subdivision_feel(onsets, beats)
    dotted = ts.beat_quarters in (1.5, 0.75)
    if feel == "threes" and not dotted:
        print(f"      ! beats subdivide in THREES (x{tri:.1f} vs x{dup:.1f} chance, {n} off-beat hits):"
              f" compound meter or a shuffle. If it's 6/8 or 12/8, rerun with --time-sig 6/8")
    elif feel == "twos" and dotted:
        n = hits_per_pulse(onsets, beats)
        bpm = 60.0 / float(np.median(np.diff(beats)))
        print(f"      ! beats subdivide in TWOS, so the tracker isn't following dotted "
              f"quarters (tracked pulse ~{bpm:.0f} bpm, hit grid {n or '?'} per pulse).")
        if n == 1 or (n == 2 and bpm >= 160):
            print("        Most likely it's following EIGHTH notes. Rerun with --beat-unit e")
        elif n in (2, 4):
            print("        Most likely it's following QUARTER notes (6/8 heard as 3/4). "
                  "Rerun with --beat-unit q")
        else:
            print("        If that pulse is a quarter note use --beat-unit q; "
                  "if an eighth, --beat-unit e")
    else:
        print(f"      subdivision feel: {feel} (threes x{tri:.1f}, twos x{dup:.1f})")
    if backbeat_between_beats(onsets, beats):
        print("      ! snares mostly land HALFWAY between tracked beats: the tracker is at "
              "half tempo (adjust --min-bpm/--max-bpm), or this is 6/8 heard as 3/4 "
              "(try --time-sig 6/8 --beat-unit q)")

    print("[5/6] mapping onsets to pro-drum notes")
    notes = onsets_to_notes(onsets, time_to_tick, snap=not args.no_snap, strengths=strengths)

    print("[6/6] writing notes.chart")
    meta = Meta(title=args.title, artist=args.artist, album=args.album,
                year=str(args.year), charter=args.charter)
    chart_path = os.path.join(args.outdir, "notes.chart")
    if os.path.exists(chart_path):   # never silently clobber a hand-corrected chart
        import time
        bak = f"{chart_path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
        shutil.copy2(chart_path, bak)
        print(f"      existing chart backed up -> {os.path.basename(bak)}")
    write_chart(chart_path, meta, tempo_events, ts, notes)

    try:
        import soundfile as sf
        length_ms = sf.info(padded_wav).duration * 1000
    except Exception:
        length_ms = None
    print("      " + write_song_ini(args.outdir, meta, tags.get("track", ""), args.genre,
                                    length_ms, args.difficulty))
    if has_cover and extract_album_art(args.input_wav, args.outdir):
        print("      album.png extracted from the input file's cover art")

    if args.lyrics or args.lyrics_file:
        print("[+]   lyrics")
        try:
            import ch_lyrics
            text = None
            if args.lyrics_file:
                with open(args.lyrics_file, encoding="utf-8-sig") as fh:
                    text = fh.read()
            dur = ch_lyrics.audio_duration(padded_wav, LEADING_SILENCE_S)
            print("      " + ch_lyrics.add_lyrics(chart_path, args.artist, args.title,
                                                  args.album or None, dur, text,
                                                  pad_s=LEADING_SILENCE_S,
                                                  sections=not args.no_sections))
        except Exception as e:     # lyrics must never sink a finished drum chart
            print(f"      ! lyrics failed ({e}); drum chart is unaffected")

    print(f"\nDone. Open {args.outdir}/notes.chart in Moonscraper to check the drafted")
    print("sections, fix tom/cymbal calls, and fine-tune syllable timing within each lyric line.")


if __name__ == "__main__":
    main()
