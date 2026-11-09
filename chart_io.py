"""
chart_io.py — minimal .chart reading/patching shared by ch_lyrics.py and
eval_thresholds.py. Edits are surgical (one block replaced) so everything else
in the file — other instruments, your hand-placed sections, star power — is
preserved byte-for-byte, including line endings.
"""

import re
from bisect import bisect_right

_SECTION_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")


def read_text(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return fh.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def parse_sections(text):
    """{section_name: [stripped body lines]} (first occurrence of a name wins)."""
    sections, cur, body = {}, None, None
    for raw in text.splitlines():
        line = raw.strip()
        m = _SECTION_RE.match(line)
        if m:
            cur, body = m.group("name"), []
            continue
        if cur is None or line == "{":
            continue
        if line == "}":
            sections.setdefault(cur, body)
            cur = None
            continue
        if line:
            body.append(line)
    return sections


def song_props(sections):
    props = {}
    for line in sections.get("Song", []):
        if "=" in line:
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip().strip('"')
    return props


def sync_tempos(sections):
    """Sorted [(tick, bpm)] from [SyncTrack] B events (defaults to 120 at 0)."""
    tempos = []
    for line in sections.get("SyncTrack", []):
        m = re.match(r"(\d+)\s*=\s*B\s+(\d+)", line)
        if m:
            tempos.append((int(m.group(1)), int(m.group(2)) / 1000.0))
    tempos.sort()
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 120.0))
    return tempos


class TempoMap:
    """tick <-> seconds for a .chart tempo map (chart time, before Offset)."""

    def __init__(self, tempos, resolution):
        self.res = resolution
        self.ticks = [t for t, _ in tempos]
        self.bpms = [b for _, b in tempos]
        self.secs = [0.0]
        for i in range(1, len(tempos)):
            dt = (self.ticks[i] - self.ticks[i - 1]) / self.res * 60.0 / self.bpms[i - 1]
            self.secs.append(self.secs[-1] + dt)

    def tick_to_sec(self, tick):
        i = max(0, bisect_right(self.ticks, tick) - 1)
        return self.secs[i] + (tick - self.ticks[i]) / self.res * 60.0 / self.bpms[i]

    def sec_to_tick(self, s):
        i = max(0, bisect_right(self.secs, s) - 1)
        return self.ticks[i] + (s - self.secs[i]) * self.bpms[i] / 60.0 * self.res


def drum_hits(sections, track="ExpertDrums"):
    """Sorted unique [(tick, class)] with pro-drum cymbal markers resolved.
    Class vocabulary matches ADTOF: kick, snare, tom, hihat, cymbal.
    Yellow cymbal -> hihat; blue/green cymbal -> cymbal (ADTOF merges ride+crash);
    any un-flagged yellow/blue/green -> tom. Accent/ghost markers are ignored."""
    by_tick = {}
    for line in sections.get(track, []):
        m = re.match(r"(\d+)\s*=\s*N\s+(\d+)\s+\d+", line)
        if m:
            by_tick.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    out = set()
    for tick, vals in by_tick.items():
        for f in vals:
            if f in (0, 32):
                cls = "kick"
            elif f == 1:
                cls = "snare"
            elif f in (2, 3, 4):
                if {2: 66, 3: 67, 4: 68}[f] in vals:
                    cls = "hihat" if f == 2 else "cymbal"
                else:
                    cls = "tom"
            elif f == 5:            # 5-lane orange
                cls = "cymbal"
            else:
                continue
            out.add((tick, cls))
    return sorted(out)


def replace_block(text, name, body_lines):
    """Replace the body of [name] { ... } with body_lines (inserting the block
    after [SyncTrack] if absent). Preserves the file's line-ending style."""
    nl = "\r\n" if "\r\n" in text else "\n"
    body = "".join(f"  {l}{nl}" for l in body_lines)
    pat = re.compile(r"(^\[" + re.escape(name) + r"\][ \t]*\r?\n[ \t]*\{[ \t]*\r?\n)"
                     r"(.*?)(^[ \t]*\}[ \t]*$)", re.M | re.S)
    m = pat.search(text)
    if m:
        return text[:m.start(2)] + body + text[m.start(3):]
    block = f"[{name}]{nl}{{{nl}{body}}}{nl}"
    sync = re.compile(r"^\[SyncTrack\][ \t]*\r?\n[ \t]*\{.*?^[ \t]*\}[ \t]*\r?\n", re.M | re.S)
    sm = sync.search(text)
    if sm:
        return text[:sm.end()] + block + text[sm.end():]
    return text + ("" if text.endswith(nl) else nl) + block
