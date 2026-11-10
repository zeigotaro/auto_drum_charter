# auto_drum_charter

Turns an audio file into a first-draft **Clone Hero pro-drums chart**: notes, tempo map, lyrics, section markers, `song.ini` and album art. The draft is meant to be opened in [Moonscraper](https://github.com/FireFox2000000/Moonscraper-Chart-Editor) and corrected by hand, not played as-is.

Drum transcription comes from [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch), which this repo is built on and still contains (see [the model](#adtof-pytorch-the-transcription-model) below).

## Quick start

```bash
python chart_drums.py "in/01 - Happiness Is All the Rage.flac" --beat-backend madmom --lyrics
```

Title, artist, album, year, genre and track number are read from the file's tags, and the song folder is named `Artist - Title`. Set `CHART_CHARTER` once so you don't have to pass `--charter`:

```powershell
setx CHART_CHARTER "YourName"
```

Anything the tags get wrong can be overridden:

```bash
python chart_drums.py song.wav -o "Artist - Title" --title "Title" --artist "Artist" --album "Album" --year 1998 --charter "YourName" --difficulty 3 --tempo-mode follow --beat-backend madmom --lyrics
```

## What you get

| File | Contents |
| --- | --- |
| `notes.chart` | Expert pro-drums notes, tempo map, lyrics and section markers. An existing chart is backed up first, never overwritten silently. |
| `song.ogg` | The full mix with 5 seconds of leading silence. |
| `song.ini` | Song info for the Clone Hero song list. Never overwrites an existing one. |
| `album.png` | Cover art, if the input file has it embedded. |
| `lyrics.txt` | Syllabified lyrics in Moonscraper's paste format. |
| `drums.ogg` | Isolated drum stem, only if Demucs is installed. |
| `_work/` | Cached intermediates (padded audio, model activations). |

## What the pipeline does

1. Pads the audio and encodes `song.ogg`.
2. Transcribes kick, snare, tom, hi-hat and cymbal from the full mix with ADTOF.
3. Drops the weakest hit wherever three or more hand drums fire at once (a drummer has two hands).
4. Tracks beats and builds the tempo map (`--tempo-mode follow` for a drummer without a click, `constant` for click-track songs).
5. Snaps notes to the grid, beat by beat: 16ths by default, triplets or 32nds only where the hits clearly sit there. `--no-snap` keeps raw timing.
6. Maps to pro-drums lanes: hi-hat on yellow cymbal, toms on blue, cymbals on green (with `--split-cymbals`, ride on blue and crash on green).
7. With `--lyrics`: fetches lyrics from [LRCLIB](https://lrclib.net), syllabifies them, and places each line at its timestamp. Syllables are spread inside the line, so line starts are real and syllable positions are estimates.
8. With `--lyrics`: drafts section markers from the stanzas (repeated stanzas become `Chorus N`, the rest `Verse N`, long lyric-free stretches become `Intro` / `Instrumental` / `Outro`), each on the nearest barline. `--no-sections` turns this off.

Not attempted: velocity, accents and ghost notes, double kick, and telling toms apart.

## What is left to do by hand

- Check the drafted sections: rename, nudge, and delete extras. `Instrumental` and `Outro` markers are the least reliable, and `Bridge`, `Pre-chorus` and `Solo` are never drafted.
- Nudge syllables within each lyric line.
- Correct the drum chart: tom lanes, ride vs crash, hi-hat density, missed and extra notes.
- Set the difficulty and preview time in `song.ini` if you didn't pass `--difficulty`.

If LRCLIB has no timestamps for the recording, lyrics fall back to evenly spread placeholders and no sections are drafted.

## Useful options

| Option | Use it when |
| --- | --- |
| `--time-sig 6/8`, `--beat-unit q\|e\|dq` | The song isn't in 4/4, or the tracker follows the wrong note value. The run prints a diagnosis and the flag to try. |
| `--downbeat-shift N` | Barlines are off by N beats. |
| `--min-bpm`, `--max-bpm`, `--transition-lambda` | madmom locks onto half or double time, or can't follow a drifting drummer. |
| `--adtof-thresholds hihat=0.35,tom=0.30` | A drum class has too many or too few notes. |
| `--auto-thresholds` | You want per-song thresholds picked from the song's own peak distribution. |
| `--reuse-activations` | Re-running the same song with different thresholds; skips the model. |
| `--note-shift-ms -50` | Notes consistently feel late or early. |
| `--split-cymbals`, `--ride-bias` | You want ride and crash separated. Needs Demucs and `audio-separator`. |
| `--lyrics-file lyrics.txt` | LRCLIB has no lyrics. A timed `.lrc` file works too. |
| `--no-stem-ogg` | Skip the drum stem. |

`python chart_drums.py -h` lists everything.

## Lyrics on an existing chart

`ch_lyrics.py` works on any chart, reading artist and title from its `[Song]` block:

```bash
python ch_lyrics.py "Artist - Title"                  # add lyrics (and sections if the chart has none)
python ch_lyrics.py "Artist - Title" --replace        # redo existing lyrics
python ch_lyrics.py "Artist - Title" --shift-ms 300   # source recording starts 0.3 s earlier than yours
python ch_lyrics.py "Artist - Title" --no-timestamps  # old behaviour: evenly spread placeholders
python ch_lyrics.py "Artist - Title" --txt-only       # just write lyrics.txt
```

Sections you placed yourself are never replaced.

## Learning thresholds from your finished charts

```bash
python eval_thresholds.py "Artist - Title" "Another Artist - Another Title" ...
```

Point it at song folders you have already corrected. It reports how many edits each threshold strategy would have cost you per drum class, learns better thresholds from your charts, and measures the systematic timing lag (a value for `--note-shift-ms`).

## Installation

```bash
pip install -e .
pip install pyphen cmudict nltk soundfile   # lyrics
pip install madmom                          # better beat tracking for loose timing (optional)
```

`ffmpeg` and `ffprobe` must be on `PATH`.

Optional: `demucs` for `drums.ogg`, and `audio-separator` (plus Demucs) for `--split-cymbals`. Both need a torch build that matches your install; a GPU is strongly advised.

Song folders, the `in/` folder and `.7z` archives are ignored by git.

## ADTOF PyTorch (the transcription model)

Automatic transcription of drum audio to MIDI, using minimal dependencies (torch, librosa, pretty_midi).

This repo is a pytorch port of [ADTOF](https://github.com/MZehren/ADTOF), by Zehren et al. described in these papers:

- M. Zehren, M. Alunno, and P. Bientinesi, “ADTOF: A large dataset of non-synthetic music for automatic drum transcription,” in Proceedings of the 22st International Society for Music Information Retrieval Conference, Online, 2021, pp. 818–824.
- Zehren, M.; Alunno, M.; Bientinesi, P. High-Quality and Reproducible Automatic Drum Transcription from Crowdsourced Data. Signals 2023, 4, 768-787. https://doi.org/10.3390/signals4040042

### Performance comparison on [MDBDrums++](https://github.com/xavriley/MDBDrumsPlusPlus) dataset

| Method | Recall | Precision | F-measure |
| --- | --- | --- | --- |
| ADTOF (original) | 88.68 | 89.90 | 88.74 |
| ADTOF-pytorch | 88.26 | 89.83 | 88.51 |

### Converting the Keras weights to Pytorch

The original implementation was in Keras/Tensorflow and used madmom for processing the audio into mel-spectrograms. Here we reimplement the model in Pytorch and convert the weights directly from the offically released weights (see `convert_weights.py`).

An exact conversion of weights between the two methods seems to be difficult, however a layer-wise comparison shows that the newer Pytorch model is close to the original:

```
Comparison with original Keras model:
  Shapes TF (1, 100, 5) vs PT (1, 100, 5)
  MAE: 0.000254
  MSE: 0.000000
  Max |diff|: 0.002278

Layer-by-layer comparison:
  sequential: MAE=0.027325 MSE=0.008597 MAX=0.581940 shape_tf=(1, 100, 10, 64) shape_pt=(1, 100, 10, 64)
  reshape: MAE=0.027325 MSE=0.008597 MAX=0.581940 shape_tf=(1, 100, 640) shape_pt=(1, 100, 640)
  bidirectional_0: MAE=0.002568 MSE=0.000273 MAX=0.266914 shape_tf=(1, 100, 120) shape_pt=(1, 100, 120)
  bidirectional_1: MAE=0.006298 MSE=0.000103 MAX=0.091772 shape_tf=(1, 100, 120) shape_pt=(1, 100, 120)
  bidirectional_2: MAE=0.005389 MSE=0.000080 MAX=0.070519 shape_tf=(1, 100, 120) shape_pt=(1, 100, 120)
  output: MAE=0.000254 MSE=0.000000 MAX=0.002278 shape_tf=(1, 100, 5) shape_pt=(1, 100, 5)

CNN detailed comparisons:
  block0_conv1_act: MAE=0.000000 MSE=0.000000 MAX=0.000000 shape_tf=(1, 100, 84, 32) shape_pt=(1, 100, 84, 32)
  block0_bn1: MAE=0.000000 MSE=0.000000 MAX=0.000020 shape_tf=(1, 100, 84, 32) shape_pt=(1, 100, 84, 32)
  block0_conv2_act: MAE=0.000002 MSE=0.000000 MAX=0.000091 shape_tf=(1, 100, 84, 32) shape_pt=(1, 100, 84, 32)
  block0_bn2: MAE=0.000005 MSE=0.000000 MAX=0.000595 shape_tf=(1, 100, 84, 32) shape_pt=(1, 100, 84, 32)
  block0_pool: MAE=0.000009 MSE=0.000000 MAX=0.000595 shape_tf=(1, 100, 28, 32) shape_pt=(1, 100, 28, 32)
  block1_conv1_act: MAE=0.000012 MSE=0.000000 MAX=0.000433 shape_tf=(1, 100, 28, 64) shape_pt=(1, 100, 28, 64)
  block1_bn1: MAE=0.000008 MSE=0.000000 MAX=0.000317 shape_tf=(1, 100, 28, 64) shape_pt=(1, 100, 28, 64)
  block1_conv2_act: MAE=0.000013 MSE=0.000000 MAX=0.000256 shape_tf=(1, 100, 28, 64) shape_pt=(1, 100, 28, 64)
  block1_bn2: MAE=0.000003 MSE=0.000000 MAX=0.000071 shape_tf=(1, 100, 28, 64) shape_pt=(1, 100, 28, 64)
  block1_pool: MAE=0.027325 MSE=0.008597 MAX=0.581940 shape_tf=(1, 100, 10, 64) shape_pt=(1, 100, 10, 64)
```

### Audio pre-processing without madmom dependency

In addition, the audio preprocessing code is reimplemented to remove the dependency on madmom which is a large, general purpose audio processing library. Again, an exact conversion was prevented as librosa and madmom use different underlying FFT implementations but the outputs appear to be sufficiently close for transcription to work:

```
Shape match: True
N bins match: True
MSE: 0.001785
Max diff: 0.976008
```

### Notes
- Model weights are bundled and loaded by default; you can override with `--weights`.
- Debug/visualization scripts are in `examples/` and require `matplotlib` (install via `[dev]`).

### Installation

Clone this repo then `pip install -e .` or `pip install -e .[dev]`

### Usage

#### CLI

```bash
adtof --audio input.wav --out output.mid \
  --thresholds 0.22,0.24,0.32,0.22,0.30 --fps 100 --device cuda
```

#### Programmatic API

```python
from adtof_pytorch import transcribe_to_midi

transcribe_to_midi("input.wav", "output.mid")
```