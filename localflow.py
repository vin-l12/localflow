#!/usr/bin/env python3
"""
localflow — private, fully-local voice dictation for macOS.

Hold the hotkey (default: Right Option), speak, release.
Your words are transcribed on-device and pasted into whatever app has focus.
No audio ever leaves this machine.

Usage:
    ./localflow.py                 run the dictation daemon
    ./localflow.py --test FILE.wav transcribe a wav file and print (no paste)
"""

import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------- config ---
HOTKEY = "alt_r"            # pynput key name: alt_r, cmd_r, ctrl_r, f13 ...
MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
SAMPLE_RATE = 16000
MIN_SECONDS = 0.3           # ignore accidental taps shorter than this
SOUNDS = True               # subtle audio cue on record start/stop
SOUND_VOLUME = "0.2"
RESTORE_CLIPBOARD = True    # put your old clipboard back after pasting
RESTORE_DELAY = 0.6         # seconds to wait before restoring
DICTIONARY_FILE = Path(__file__).parent / "dictionary.json"
# ----------------------------------------------------------------------------

import numpy as np
import sounddevice as sd


def load_model():
    """Load Parakeet. If the model is already on disk, force offline mode
    BEFORE the library is imported (it reads the flag at import time), so
    startup never touches the network."""
    import os

    cache = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--"))
    if cache.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("model not cached yet — downloading once (~1.2 GB) ...")

    from parakeet_mlx import from_pretrained
    return from_pretrained(MODEL)


def load_dictionary():
    if not DICTIONARY_FILE.exists():
        return []
    entries = json.loads(DICTIONARY_FILE.read_text())
    return [
        (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
        for k, v in entries.items()
    ]


def apply_dictionary(text, rules):
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def play(sound):
    if SOUNDS:
        subprocess.Popen(
            ["afplay", "-v", SOUND_VOLUME, f"/System/Library/Sounds/{sound}.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def get_clipboard():
    r = subprocess.run(["pbpaste"], capture_output=True)
    return r.stdout


def set_clipboard(data):
    subprocess.run(["pbcopy"], input=data)


def paste_text(text):
    """Deliver text to the frontmost app via clipboard + Cmd-V."""
    old = get_clipboard() if RESTORE_CLIPBOARD else None
    set_clipboard(text.encode())
    r = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("  ⚠ macOS blocked the paste — your words are on the clipboard,")
        print("    press Cmd-V yourself to paste them.")
        print("    Fix: System Settings → Privacy & Security → Accessibility →")
        print("    turn ON your terminal app, then restart localflow.")
        print(f"    (macOS said: {r.stderr.strip()})")
        return
    if old is not None:
        time.sleep(RESTORE_DELAY)
        set_clipboard(old)


class Recorder:
    """Opens the mic only while the hotkey is held (no always-on orange dot)."""

    def __init__(self):
        self.frames = []
        self.stream = None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda data, *_: self.frames.append(data.copy()),
        )
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.frames)[:, 0]


class LocalFlow:
    def __init__(self):
        print("localflow · loading model ...")
        self.model = load_model()
        self.rules = load_dictionary()
        self.recorder = Recorder()
        self.recording = False
        self.jobs = queue.Queue()
        threading.Thread(target=self.worker, daemon=True).start()

    def transcribe(self, audio):
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(mx.array(audio), self.model.preprocessor_config)
        return self.model.generate(mel)[0].text.strip()

    def worker(self):
        while True:
            audio = self.jobs.get()
            t0 = time.time()
            text = self.transcribe(audio)
            if not text:
                print("  (heard nothing)")
                continue
            text = apply_dictionary(text, self.rules)
            paste_text(text)
            print(f"  {time.time() - t0:.1f}s · {text}")

    # -- hotkey handlers ----------------------------------------------------
    def on_press(self, key):
        if key == self.hotkey and not self.recording:
            self.recording = True
            self.recorder.start()
            play("Pop")

    def on_release(self, key):
        if key == self.hotkey and self.recording:
            self.recording = False
            audio = self.recorder.stop()
            play("Bottle")
            if len(audio) / SAMPLE_RATE >= MIN_SECONDS:
                self.jobs.put(audio)

    def run(self):
        from pynput import keyboard

        self.hotkey = getattr(keyboard.Key, HOTKEY)
        # warm up so the first real dictation is fast
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        print(f"ready · hold {HOTKEY} to dictate, release to paste · ctrl-c quits")
        print("  (nothing appearing? grant Input Monitoring + Accessibility ")
        print("   to your terminal in System Settings → Privacy & Security)")
        with keyboard.Listener(on_press=self.on_press,
                               on_release=self.on_release) as listener:
            listener.join()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--test":
        import soundfile as sf

        app = LocalFlow()
        audio, sr = sf.read(sys.argv[2], dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz wav, got {sr}"
        print(apply_dictionary(app.transcribe(audio), app.rules))
        return
    try:
        LocalFlow().run()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
