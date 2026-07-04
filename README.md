# localflow

Private, fully-local voice dictation for macOS — an open-source alternative
to Wispr Flow where **no audio ever leaves the machine**.

Wispr Flow is cloud-only by design: every dictation is streamed to their
servers (Baseten, OpenAI, Anthropic, Cerebras, AWS) and even its Privacy Mode
only disables retention, not the upload. localflow reproduces the core UX —
hold a key, speak, release, text appears where your cursor is — with the
speech model running on this Mac's GPU.

## Architecture

```
hold Right-Option ──► mic capture (sounddevice, 16 kHz mono)
release           ──► Parakeet TDT 0.6B on Apple MLX (Metal GPU, on-device)
                  ──► custom dictionary fix-ups (dictionary.json)
                  ──► clipboard + synthetic Cmd-V into the frontmost app
                        (old clipboard restored afterwards)
```

- **Model**: `mlx-community/parakeet-tdt-0.6b-v2` — NVIDIA Parakeet, the same
  family the good offline dictation apps use. Punctuates, capitalises, and
  formats numbers ("four hundred dollars" → `$400`). ~1 s warm latency on the
  M2 Pro, ~1.2 GB on disk, downloaded once then loaded with `HF_HUB_OFFLINE=1`.
- **Mic is only open while the key is held** — no always-on orange dot.
- Transcription runs on a worker thread, so you can start the next dictation
  while the last one is still processing.

## Requirements

- Apple Silicon Mac (M1 or later — the model runs on the GPU via MLX)
- Python 3.10+ (`brew install python` if you only have the system 3.9)

## Setup & run

```sh
git clone https://github.com/vin-l12/localflow && cd localflow
./LocalFlow.command      # or double-click it in Finder
```

First launch creates the environment, downloads the model (~1.2 GB, once),
then says `ready`. After that it's fully offline. macOS will ask for three
permissions, granted to **your terminal app** (Terminal/iTerm/etc):

1. **Input Monitoring** — to see the global hotkey (may require reopening
   the terminal once).
2. **Microphone** — prompted on your first dictation.
3. **Accessibility** — to press Cmd-V for you (System Events prompt).

Then hold **Right Option**, speak, release. Quiet pop = recording, bottle =
processing, text lands at your cursor ~1 s later.
`.venv/bin/python localflow.py --test file.wav` transcribes a 16 kHz wav and
prints instead of pasting (pipeline verification).

To start it automatically at login: System Settings → General → Login Items
→ **+** → add `LocalFlow.command`.

## Config

Constants at the top of `localflow.py`: hotkey, model, sounds, clipboard
restore. `dictionary.json` maps misheard words to your spelling
(case-insensitive whole-word), e.g. `"vwap": "VWAP"`.

## Not built (ideas, not promises)

Menu-bar app instead of a terminal window · streaming partial transcripts ·
tone/context rewriting via a local LLM (Ollama) · per-app formatting.
