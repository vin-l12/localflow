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
release           ──► Whisper large-v3-turbo on Apple MLX (Metal GPU, on-device)
                        (biased toward dictionary.json vocab via initial_prompt)
                  ──► custom dictionary fix-ups (dictionary.json)
                  ──► LLM cleanup pass (llama3.2:3b via local Ollama):
                        fixes mis-heard words from context, strips "um"s and
                        false starts — skipped gracefully if Ollama is down
                  ──► clipboard + synthetic Cmd-V into the frontmost app
                        (old clipboard restored afterwards)
```

- **Model**: `mlx-community/whisper-large-v3-turbo` — OpenAI's Whisper
  large-v3-turbo, ~1.6 GB on disk, downloaded once then loaded with
  `HF_HUB_OFFLINE=1`. Punctuates, capitalises, and formats numbers
  ("four hundred dollars" → `$400`). The previous engine, NVIDIA Parakeet
  TDT 0.6B (`AB_MODEL` in the config block), is kept around for comparisons:
  `./localflow.py --ab file.wav` transcribes with both and prints both, and
  `./localflow.py --record file.wav` records a mic sample to compare on.
  Setting `MODEL` back to the Parakeet id swaps the engine.
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

First launch creates the environment, downloads the model (~1.6 GB, once),
then says `ready`. After that it's fully offline. macOS will ask for three
permissions, granted to **your terminal app** (Terminal/iTerm/etc):

1. **Input Monitoring** — to see the global hotkey (may require reopening
   the terminal once).
2. **Microphone** — prompted on your first dictation.
3. **Accessibility** — to press Cmd-V for you (System Events prompt).

Then hold **Right Option**, speak, release — or **tap it once** to start,
speak hands-free, and tap again to stop (a take also ends itself after 10 s
of silence). Quiet pop = recording, bottle = processing, text lands at your
cursor ~1 s later.

The mic is pinned by name (`INPUT_DEVICE` in the config block) rather than
following the system default — otherwise an iPhone/AirPods continuity mic can
silently take over the input and every dictation records the wrong room.

> Change the key via `HOTKEY` in `localflow.py` (`alt_r`, `cmd_r`, `ctrl_r`,
> `f13`). Note: the Fn/Globe key is *not* supported — macOS sets the same
> "function" modifier flag for the arrow and F-keys, so it can't be told apart
> from them without a dedicated event tap.

`.venv/bin/python localflow.py --test file.wav` transcribes a 16 kHz wav and
prints instead of pasting (pipeline verification).

## Run as a menu-bar app (no terminal)

```sh
./make_app.sh
```

builds **`~/Applications/LocalFlow.app`** — launch it from Spotlight or
Finder like any app. No terminal window: a small ◌ appears in the menu bar,
turning ◎ when the model is warm, with a **Quit localflow** menu item.
Output goes to `~/Library/Logs/localflow.log`.

macOS treats the bundle as a new app, so grant the same three permissions
again — this time to **LocalFlow.app** (use **+** in each pane if it isn't
listed), then relaunch. Re-run `make_app.sh` if you move this folder.

To start it automatically at login: System Settings → General → Login Items
→ **+** → add `LocalFlow.app` (or `LocalFlow.command` for the terminal
version).

## LLM cleanup pass (optional, recommended)

A second on-device pass through a small LLM makes transcripts read the way
you *meant* them: homophones fixed from context, fillers and false starts
dropped, punctuation cleaned. Setup (free, local, ~2 GB once):

```sh
brew install ollama
brew services start ollama     # keeps it running at login
ollama pull llama3.2:3b
```

Costs ~0.5–1 s per dictation. If Ollama isn't running, localflow notices and
pastes the raw transcript instead — dictation never blocks on it. Disable
with `CLEANUP = False` in the config block.

## Config

Constants at the top of `localflow.py`: hotkey, model, sounds, clipboard
restore, cleanup. `dictionary.json` maps misheard words to your spelling
(case-insensitive whole-word), e.g. `"vwap": "VWAP"` — entries also bias
what Whisper hears in the first place.

## Not built (ideas, not promises)

Streaming partial transcripts · a learning loop that suggests dictionary
entries from your transcript history · per-app formatting.
