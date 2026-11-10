#!/usr/bin/env python3
"""
ch_lyrics.py — fetch lyrics for a song, clean and syllabify them, and add them
to a Clone Hero .chart, ready for Moonscraper's lyric editor. When the source has
line timestamps (LRCLIB synced lyrics or an .lrc file) each phrase is placed at
its real time and section markers are drafted; otherwise syllables are untimed
placeholders.

Works standalone on any existing chart (artist/title read from its [Song] block):

    python ch_lyrics.py "out/Ride - Leave Them All Behind"
    python ch_lyrics.py "out/.../notes.chart" --replace          # redo existing lyrics
    python ch_lyrics.py "out/.../notes.chart" --lyrics-file mine.txt   # skip the download

...or is called by chart_drums.py with --lyrics.

Outputs:
  lyrics.txt   one phrase per line, syllables joined by dashes ("Hel-lo dar-ling"),
               the paste format used by Moonscraper's lyric editor / Lyric Adder.
  notes.chart  [Events] gets phrase_start / "lyric <syl>" / phrase_end events,
               merged with (never replacing) your sections. With timestamps,
               each phrase starts at its line time and syllables are spread
               inside the line; without, syllables are spread evenly across the
               drum-note span as PLACEHOLDERS. If the chart has no sections,
               "section ..." events are drafted from the stanzas.

Source: LRCLIB (lrclib.net) — free, no API key. Tries an exact-signature match
(artist + title + duration) first, then falls back to search, preferring the
result whose duration is closest to your audio.

Syllabification: pyphen hyphenation, validated against the CMU Pronouncing
Dictionary's syllable count, with a sonority-based fallback when they disagree.
On a hand-labelled test set this gets ~97% of syllable COUNTS right (count is
what matters for one-event-per-sung-syllable); boundary placement is sometimes
arguable ("ci-ty" vs "cit-y"). Words not in CMU are trusted to pyphen.

Dependencies: pip install pyphen cmudict nltk
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import chart_io

LRCLIB = "https://lrclib.net/api"
# LRCLIB asks clients to identify themselves; keep this descriptive.
USER_AGENT = "chart_drums-lyrics/1.0 (personal Clone Hero charting tool)"
DEFAULT_PAD_S = 5.0     # chart_drums.py's leading silence baked into song.ogg

# --------------------------------------------------------------------------- #
# Fetch                                                                       #
# --------------------------------------------------------------------------- #

def _http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return None
        raise


_LRC_TAG = re.compile(r"^\s*(\[\d+:\d+(?:[.:]\d+)?\]\s*)+", re.M)


def _text_of(rec):
    """Plain lyrics from an LRCLIB record; falls back to synced lyrics with
    timestamps stripped (records can have plainLyrics null but synced present)."""
    if not rec or rec.get("instrumental"):
        return None
    if rec.get("plainLyrics"):
        return rec["plainLyrics"]
    if rec.get("syncedLyrics"):
        return _LRC_TAG.sub("", rec["syncedLyrics"])
    return None


_LRC_LINE = re.compile(r"^\s*((?:\[\d+:\d+(?:[.:]\d+)?\]\s*)+)(.*)$")
_LRC_TIME = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")


def parse_lrc(text):
    """LRC text -> sorted [(seconds, line)] or None if it isn't (mostly) timed.
    Empty timed lines are kept: they mark where the previous line stops."""
    timed, untimed = [], 0
    for line in (text or "").splitlines():
        m = _LRC_LINE.match(line)
        if not m:
            if line.strip() and not re.match(r"^\s*\[\w+:.*\]\s*$", line):
                untimed += 1
            continue
        for mm, ss in _LRC_TIME.findall(m.group(1)):
            timed.append((int(mm) * 60 + float(ss.replace(":", ".")), m.group(2).strip()))
    if len(timed) < 2 or untimed > len(timed) // 4:
        return None
    timed.sort(key=lambda r: r[0])
    return timed


def _synced_of(rec):
    if not rec or rec.get("instrumental"):
        return None
    return parse_lrc(rec.get("syncedLyrics") or "")


def fetch_lyrics(artist, title, album=None, duration=None):
    """Return (text, how, synced) or (None, reason, None). synced is
    [(seconds, line)] in the SOURCE recording's time (no leading pad), or None."""
    q = {"artist_name": artist, "track_name": title}
    if album:
        q["album_name"] = album
    if duration:
        q["duration"] = int(round(duration))
    try:
        rec = _http_json(f"{LRCLIB}/get?{urllib.parse.urlencode(q)}")
    except urllib.error.URLError as e:
        return None, f"network error reaching LRCLIB ({e.reason})", None
    if rec and rec.get("instrumental"):
        return None, "LRCLIB marks this track as instrumental", None
    text = _text_of(rec)
    if text and _synced_of(rec):
        return text, "exact match", _synced_of(rec)
    exact = text

    try:
        results = _http_json(f"{LRCLIB}/search?" + urllib.parse.urlencode(
            {"artist_name": artist, "track_name": title})) or []
    except urllib.error.URLError as e:
        if exact:
            return exact, "exact match", None
        return None, f"network error reaching LRCLIB ({e.reason})", None
    cands = [r for r in results if _text_of(r)]
    if duration:
        # A timed record only helps if it's the same recording: require its
        # length within 2s of ours, else its timestamps would be misleading.
        diff = lambda r: abs((r.get("duration") or 0) - duration)
        cands.sort(key=diff)
        timed = [r for r in cands if _synced_of(r) and diff(r) <= 2.0]
    else:
        timed = [r for r in cands if _synced_of(r)]
    if exact and not timed:
        return exact, "exact match", None
    if not cands:
        return None, "no lyrics found on LRCLIB (try --lyrics-file)", None
    best = timed[0] if timed else cands[0]
    how = f"search match: {best.get('artistName')} - {best.get('trackName')}"
    if duration and best.get("duration"):
        how += f" ({best['duration']:.0f}s vs your {duration:.0f}s)"
    return _text_of(best), how, (_synced_of(best) if timed else None)


# --------------------------------------------------------------------------- #
# Clean                                                                       #
# --------------------------------------------------------------------------- #

_HEADER = re.compile(r"\[[^\]]*\]|\{[^}]*\}")                 # [Verse 1], [Chorus: X]
_REPEAT = re.compile(r"\(\s*[x×]\s*\d+\s*\)|\b[x×]\d+\b", re.I)  # (x2), x3
_CONTROL = re.compile(r"[#^+%$/_\"“”]")                       # RB/CH lyric control chars


def clean_lyrics(text, drop_parentheticals=False):
    """-> list of phrase strings (one per non-empty line)."""
    out = []
    for line in text.splitlines():
        line = _HEADER.sub("", line)
        line = _REPEAT.sub("", line)
        if drop_parentheticals:
            line = re.sub(r"\([^)]*\)", "", line)
        line = line.replace("\u2019", "'").replace("\u2018", "'")
        line = _CONTROL.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Syllabify                                                                   #
# --------------------------------------------------------------------------- #

_VOWELS = set("aeiouyAEIOUY")


class Syllabifier:
    def __init__(self, lang="en_US"):
        try:
            import pyphen
        except ImportError:
            sys.exit("Lyrics need: pip install pyphen cmudict nltk")
        self.hy = pyphen.Pyphen(lang=lang, left=1, right=2)
        self.cmu, self.son = None, None
        if lang.lower().startswith("en"):
            try:
                import cmudict
                self.cmu = cmudict.dict()
            except ImportError:
                print("  ! cmudict not installed; syllable counts unvalidated")
            try:
                from nltk.tokenize import SyllableTokenizer
                self.son = SyllableTokenizer()
            except ImportError:
                pass

    @staticmethod
    def _merge_vowelless(parts):
        out = []
        for p in parts:
            if out and (not (set(p) & _VOWELS) or not (set(out[-1]) & _VOWELS)):
                out[-1] += p
            else:
                out.append(p)
        return out

    def _count(self, word):
        if not self.cmu:
            return None
        prons = self.cmu.get(word.lower())
        if not prons:
            return None
        return min(sum(ph[-1].isdigit() for ph in p) for p in prons)

    def _hy(self, w):
        parts = self._merge_vowelless([p for p in self.hy.inserted(w).split("-") if p])
        # pyphen splits a consonant "y" off the front (y-ou, y-our-self); a
        # leading y before a vowel never forms a syllable on its own.
        if len(parts) > 1 and parts[0].lower() == "y" and parts[1][0] in _VOWELS:
            parts[:2] = [parts[0] + parts[1]]
        return parts

    def word(self, w):
        if len(w) <= 2:
            return [w]
        pieces, n = self._hy(w), self._count(w)
        if n == 1:                     # CMU says one syllable: never split
            return [w]
        if n is None or len(pieces) == n:
            return pieces
        if self.son:
            alt = self._merge_vowelless(self.son.tokenize(w))
            if len(alt) == n:
                return alt
        # sung contractions: could-n't, did-n't, was-n't
        if len(pieces) + 1 == n and w.lower().endswith("n't") and len(pieces[-1]) > 3:
            return pieces[:-1] + [pieces[-1][:-3], pieces[-1][-3:]]
        # leading lone vowel: a-way, o-ver, e-ven, a-lone
        if len(pieces) + 1 == n and w[0] in _VOWELS and w[1] not in _VOWELS:
            rest = self._hy(w[1:])
            return [w[0]] + (rest if len(rest) == n - 1 else [w[1:]])
        return pieces

    def line(self, line):
        """-> list of syllable tokens; a trailing '-' joins to the next token."""
        syls = []
        for tok in line.split():
            # split hyphenated compounds / dashes into separate words
            for part in (p for p in re.split(r"[-\u2010-\u2015]+", tok) if p):
                m = re.match(r"^([^\w']*)(.*?)([^\w']*)$", part)
                lead, core, trail = m.groups()
                if not core:
                    continue
                pieces = self.word(core)
                pieces[0] = lead + pieces[0]          # sequential: may be the
                pieces[-1] = pieces[-1] + trail       # same element (1 syllable)
                syls += [p + "-" for p in pieces[:-1]] + [pieces[-1]]
        return syls


def phrase_text(syls):
    """Syllable tokens -> 'Hel-lo dar-ling' (Moonscraper lyric-editor paste format)."""
    return "".join(s if s.endswith("-") else s + " " for s in syls).strip()


# --------------------------------------------------------------------------- #
# Placeholder events + chart merge                                            #
# --------------------------------------------------------------------------- #

def placeholder_events(phrases, start_tick, end_tick, resolution, gap_units=2):
    """Spread syllables evenly over [start, end]; returns [(tick, event_text)]."""
    n = sum(len(p) for p in phrases) + gap_units * len(phrases)
    step = max(1, (end_tick - start_tick) // max(1, n))
    grid = resolution // 4                         # snap to 16ths when there's room
    if step >= grid:
        step = step // grid * grid
        start_tick = -(-start_tick // grid) * grid
    ev, tick = [], start_tick
    for syls in phrases:
        ev.append((tick, "phrase_start"))
        for s in syls:
            ev.append((tick, f"lyric {s}"))
            tick += step
        ev.append((tick, "phrase_end"))
        tick += step * gap_units
    return ev


def timed_events(phrases, starts, ends, tmap, resolution,
                 max_syl_s=0.45, fill=0.9):
    """Place each phrase at its real line time. phrases[i] starts at starts[i]
    seconds (chart time) and must finish by ends[i] (the next line's start).
    Syllables are spread evenly inside the line, capped at max_syl_s each so a
    line followed by an instrumental gap doesn't smear across it. Line starts
    are real; syllable positions inside the line are still estimates."""
    grid = resolution // 4                          # 16th notes
    snap = lambda s: int(round(tmap.sec_to_tick(s) / grid)) * grid
    ev, floor = [], 0
    for syls, t0, t1 in zip(phrases, starts, ends):
        n = len(syls)
        span = min(max(0.0, t1 - t0) * fill, n * max_syl_s)
        a = max(floor, snap(t0))
        b = max(a + n, snap(t0 + span))             # tick of phrase_end
        limit = snap(t1)
        if limit > a + n:
            b = min(b, limit)
        ev.append((a, "phrase_start"))
        last = a - 1
        for i, s in enumerate(syls):
            tick = a + (b - a) * i // n
            if (b - a) // n >= grid:
                tick = tick // grid * grid
            tick = max(tick, last + 1)
            ev.append((tick, f"lyric {s}"))
            last = tick
        b = max(b, last + 1)
        ev.append((b, "phrase_end"))
        floor = b
    return ev


# --------------------------------------------------------------------------- #
# Draft section markers from timed lyrics                                     #
# --------------------------------------------------------------------------- #

def chart_barlines(sections, resolution, end_tick):
    """Barline ticks from the chart's TS events (4/4 if none), up to end_tick."""
    ts = []
    for line in sections.get("SyncTrack", []):
        m = re.match(r"(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?", line)
        if m:
            den = 2 ** int(m.group(3)) if m.group(3) else 4
            ts.append((int(m.group(1)), max(1, round(resolution * 4 * int(m.group(2)) / den))))
    ts.sort()
    if not ts or ts[0][0] != 0:
        ts.insert(0, (0, resolution * 4))
    bars = []
    for i, (tick, measure) in enumerate(ts):
        stop = ts[i + 1][0] if i + 1 < len(ts) else end_tick + measure
        bars.extend(range(tick, stop, measure))
    return bars


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def group_stanzas(lines, plain_text=None, gap_s=3.0):
    """lines: [(start_s, end_s, text)] -> list of stanzas (lists of line indices).
    Uses the blank lines of the plain lyrics when they line up one-to-one with
    the timed lines; otherwise starts a new stanza at every pause >= gap_s."""
    if plain_text:
        blocks, cur = [], []
        for raw in plain_text.splitlines():
            if clean_lyrics(raw):
                cur.append(raw)
            elif cur:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        if len(blocks) > 1 and sum(map(len, blocks)) == len(lines):
            out, i = [], 0
            for b in blocks:
                out.append(list(range(i, i + len(b))))
                i += len(b)
            return out
    out = [[0]]
    for i in range(1, len(lines)):
        if lines[i][0] - lines[i - 1][1] >= gap_s:
            out.append([])
        out[-1].append(i)
    return out


def draft_sections(lines, stanzas, music_start_s, music_end_s, tmap, barlines,
                   min_gap_s=16.0, similar=0.6):
    """-> [(tick, name)]. Stanzas whose words repeat elsewhere are the Chorus
    (the most-repeated group), the rest are Verses; long lyric-free stretches
    become Intro / Instrumental / Outro. Every marker sits on the barline
    nearest the first sung line. A draft to rename and nudge, not a verdict."""
    import difflib
    from bisect import bisect_left
    texts = [_words(" ".join(lines[i][2] for i in st)) for st in stanzas]
    groups = []                                    # lists of stanza indices
    for i, w in enumerate(texts):
        for g in groups:
            if difflib.SequenceMatcher(None, texts[g[0]], w).ratio() >= similar:
                g.append(i)
                break
        else:
            groups.append([i])
    repeated = [g for g in groups if len(g) > 1]
    chorus = set(max(repeated, key=lambda g: (len(g), len(texts[g[0]])))) if repeated else set()

    def bar_near(sec):
        t = tmap.sec_to_tick(sec)
        k = bisect_left(barlines, t)
        cands = barlines[max(0, k - 1):k + 1]
        return min(cands, key=lambda b: abs(b - t))

    def bar_after(sec):
        t = tmap.sec_to_tick(sec)
        return barlines[min(len(barlines) - 1, bisect_left(barlines, t))]

    out, counts, prev_kind = [], {"Verse": 0, "Chorus": 0}, None
    first = lines[stanzas[0][0]][0]
    if first - music_start_s >= 4.0:
        out.append((bar_near(music_start_s), "Intro"))
    for i, st in enumerate(stanzas):
        start = lines[st[0]][0]
        kind = "Chorus" if i in chorus else "Verse"
        if i and start - lines[stanzas[i - 1][-1]][1] >= min_gap_s:
            out.append((bar_after(lines[stanzas[i - 1][-1]][1]), "Instrumental"))
            prev_kind = None
        if kind == prev_kind:                      # back-to-back stanzas of one kind
            continue                               # are one section (a two-part verse)
        counts[kind] += 1
        out.append((bar_near(start), f"{kind} {counts[kind]}"))
        prev_kind = kind
    last_end = lines[stanzas[-1][-1]][1]
    if music_end_s - last_end >= min_gap_s:
        out.append((bar_after(last_end), "Outro"))
    dedup = []
    for tick, name in out:                         # one marker per barline, in order
        if dedup and tick <= dedup[-1][0]:
            continue
        dedup.append((tick, name))
    return dedup


_EV_RE = re.compile(r'^(\d+)\s*=\s*E\s+"(.*)"\s*$')
_RANK = {"phrase_end": 0, "phrase_start": 2}


def _rank(text):
    if text.startswith("lyric "):
        return 3
    return _RANK.get(text, 1)          # sections and anything else: 1


def merge_events(existing_lines, new_events, replace):
    kept, had_lyrics = [], False
    for i, line in enumerate(existing_lines):
        m = _EV_RE.match(line)
        if not m:
            kept.append((10**12, 1, i, line))      # keep unknown lines, at end
            continue
        tick, text = int(m.group(1)), m.group(2)
        if text.startswith("lyric ") or text in ("phrase_start", "phrase_end"):
            had_lyrics = True
            if replace:
                continue
        kept.append((tick, _rank(text), i, line))
    if had_lyrics and not replace:
        return None
    base = len(existing_lines)
    for j, (tick, text) in enumerate(new_events):
        kept.append((tick, _rank(text), base + j, f'{tick} = E "{text}"'))
    kept.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in kept]


def audio_duration(path, pad_s):
    try:
        import soundfile as sf
        return sf.info(path).duration - pad_s
    except Exception:
        try:
            import librosa
            return librosa.get_duration(path=path) - pad_s
        except Exception:
            return None


def add_lyrics(chart_path, artist=None, title=None, album=None, duration=None,
               lyrics_text=None, lang="en_US", replace=False,
               drop_parentheticals=False, write_events=True, pad_s=DEFAULT_PAD_S,
               use_timestamps=True, shift_s=0.0, sections=True):
    """Fetch/clean/syllabify; write lyrics.txt; merge placeholder events into the
    chart. Returns a short status string. Raises nothing for 'not found'."""
    text = chart_io.read_text(chart_path)
    secs = chart_io.parse_sections(text)
    props = chart_io.song_props(secs)
    artist = artist or props.get("Artist")
    title = title or props.get("Name")
    album = album or props.get("Album") or None
    folder = os.path.dirname(os.path.abspath(chart_path))
    if duration is None:
        ogg = os.path.join(folder, props.get("MusicStream") or "song.ogg")
        if os.path.exists(ogg):
            duration = audio_duration(ogg, pad_s)

    if lyrics_text is None:
        if not artist or not title or artist == "Unknown" or title == "Unknown":
            return "skipped: need artist and title (flags or chart [Song] block)"
        lyrics_text, how, synced = fetch_lyrics(artist, title, album, duration)
        if lyrics_text is None:
            return f"skipped: {how}"
    else:
        how, synced = "from file", parse_lrc(lyrics_text)   # .lrc files work too
    if not use_timestamps:
        synced = None

    syl = Syllabifier(lang)
    starts = ends = None
    n_sections = 0
    if synced:
        # One phrase per timed line; a line ends where the next timed line
        # (even an empty one) begins.
        phrases, starts, ends, raws = [], [], [], []
        for i, (t, raw) in enumerate(synced):
            cl = clean_lyrics(raw, drop_parentheticals)
            s = syl.line(" ".join(cl)) if cl else []
            if not s:
                continue
            nxt = synced[i + 1][0] if i + 1 < len(synced) else t + 0.45 * len(s)
            phrases.append(s)
            raws.append(" ".join(cl))
            starts.append(t + pad_s + shift_s)
            ends.append(nxt + pad_s + shift_s)
    else:
        lines = clean_lyrics(lyrics_text, drop_parentheticals)
        phrases = [s for s in (syl.line(l) for l in lines) if s]
    if not phrases:
        return "skipped: lyrics were empty after cleanup"

    txt_path = os.path.join(folder, "lyrics.txt")
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(phrase_text(p) for p in phrases) + "\n")
    status = (f"{len(phrases)} phrases / {sum(map(len, phrases))} syllables "
              f"({how}) -> lyrics.txt")
    if not write_events:
        return status

    res = int(float(props.get("Resolution", 192)))
    ticks = [t for t, _ in chart_io.drum_hits(secs)]
    start, end = (min(ticks), max(ticks)) if ticks else (res * 4, res * 4 * 64)
    if synced:
        tmap = chart_io.TempoMap(chart_io.sync_tempos(secs), res)
        events = timed_events(phrases, starts, ends, tmap, res)
        has_sections = any(_EV_RE.match(l) and _EV_RE.match(l).group(2).startswith("section ")
                           for l in secs.get("Events", []))
        if sections and not has_sections:
            # a line is sung from its start until the next line, at most ~0.45s/syllable
            lines = [(a, min(b, a + 0.45 * len(p)), r)
                     for a, b, p, r in zip(starts, ends, phrases, raws)]
            plain = None if how == "from file" else lyrics_text
            music_end = tmap.tick_to_sec(end) if ticks else lines[-1][1]
            drafted = draft_sections(lines, group_stanzas(lines, plain), pad_s + shift_s,
                                     music_end, tmap, chart_barlines(secs, res, end))
            events += [(tick, f"section {name}") for tick, name in drafted]
            n_sections = len(drafted)
    else:
        events = placeholder_events(phrases, start, end, res)
    merged = merge_events(secs.get("Events", []), events, replace)
    if merged is None:
        return status + "; chart already has lyrics, left untouched (use --replace)"
    chart_io.write_text(chart_path, chart_io.replace_block(text, "Events", merged))
    if synced:
        return status + ("; LINE-TIMED events added to chart (line starts from "
                         "source timestamps; syllables spread within each line)"
                         + (f"; {n_sections} section markers drafted" if n_sections else ""))
    return status + "; untimed placeholder events added to chart"


def main():
    ap = argparse.ArgumentParser(description="Add syllabified lyrics (line-timed when the source has "
                                             "timestamps) and draft sections to a .chart")
    ap.add_argument("chart", help="notes.chart or its song folder")
    ap.add_argument("--artist"); ap.add_argument("--title"); ap.add_argument("--album")
    ap.add_argument("--lyrics-file", help="use this text file instead of downloading")
    ap.add_argument("--lang", default="en_US", help="pyphen language (CMU check is English-only)")
    ap.add_argument("--replace", action="store_true", help="replace existing lyric events")
    ap.add_argument("--drop-parentheticals", action="store_true",
                    help="remove (backing vocal) text in parentheses")
    ap.add_argument("--txt-only", action="store_true", help="write lyrics.txt, don't touch the chart")
    ap.add_argument("--pad-seconds", type=float, default=DEFAULT_PAD_S,
                    help="leading silence in song.ogg, subtracted for duration matching")
    ap.add_argument("--no-timestamps", action="store_true",
                    help="ignore LRCLIB/.lrc line timestamps; spread syllables evenly")
    ap.add_argument("--no-sections", action="store_true",
                    help="don't draft section markers (only ever added to a chart with none)")
    ap.add_argument("--shift-ms", type=float, default=0.0,
                    help="move all timed lyric lines by this many ms (+ = later), for "
                         "when the lyric source's recording starts earlier/later than yours")
    a = ap.parse_args()

    chart = os.path.join(a.chart, "notes.chart") if os.path.isdir(a.chart) else a.chart
    lyrics_text = None
    if a.lyrics_file:
        with open(a.lyrics_file, encoding="utf-8-sig") as fh:
            lyrics_text = fh.read()
    print(add_lyrics(chart, a.artist, a.title, a.album, None, lyrics_text, a.lang,
                     a.replace, a.drop_parentheticals, not a.txt_only, a.pad_seconds,
                     not a.no_timestamps, a.shift_ms / 1000.0, not a.no_sections))


if __name__ == "__main__":
    main()
