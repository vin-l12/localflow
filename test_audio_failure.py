#!/usr/bin/env python3
"""Reproduce the stuck-hotkey bug: when the mic fails to open/close (PortAudio
-9986 after long uptime), the key handlers must NOT propagate the exception
(which kills pynput's listener) and must NOT leave self.recording stuck True."""
import queue
import sys
import threading

import numpy as np

import localflow

localflow.play = lambda *_: None  # silence system sounds during the test


class FakeKey:
    """Stands in for a pynput Key — the handlers use == and .name."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, FakeKey) and other.name == self.name


HK = FakeKey("alt_r")


def bare_app():
    """A LocalFlow with only the attributes the key handlers touch — avoids
    loading the whisper model."""
    lf = localflow.LocalFlow.__new__(localflow.LocalFlow)
    lf.recording = False
    lf.finish_lock = threading.Lock()
    lf.hotkey = HK
    lf.release_keys = {HK}
    lf.jobs = queue.Queue()
    return lf


class BoomOnStart:
    def start(self):
        raise RuntimeError("Error opening InputStream: PaErrorCode -9986")

    def stop(self):
        return np.zeros(0, dtype=np.float32)


class BoomOnStop:
    def start(self):
        pass

    def stop(self):
        raise RuntimeError("Error closing InputStream: PaErrorCode -9986")


def test_failed_start_does_not_stick_or_propagate():
    lf = bare_app()
    lf.recorder = BoomOnStart()
    lf.on_press(HK)  # must not raise
    assert lf.recording is False, "recording stuck True after a failed mic open"
    # a second press must still be able to start a take
    lf.recorder = BoomOnStart()
    lf.on_press(HK)
    assert lf.recording is False
    print("PASS: failed mic open — no crash, recording reset")


def test_failed_stop_does_not_stick_or_propagate():
    lf = bare_app()
    lf.recorder = BoomOnStop()
    lf.recording = True  # a take is in progress
    lf.on_release(HK)  # must not raise even though stop() throws
    assert lf.recording is False, "recording stuck True after a failed mic close"
    print("PASS: failed mic close — no crash, recording reset")


class DeadStream:
    """The wedge that ends sessions: the stream OPENS without error but the
    audio callback never fires — 15s of speech comes back as zero frames."""

    def __init__(self, held_seconds=15.0):
        import time
        self.t0 = time.time() - held_seconds  # take was held this long
        self.frames = []

    def start(self):
        pass  # opens "fine" — that is the whole problem

    def stop(self):
        return np.zeros(0, dtype=np.float32)  # ...but delivered nothing


def test_wedged_close_never_blocks_and_keeps_the_audio():
    """THE REPRODUCED HANG (Jul 16): Pa_StopStream never returned on the 13th
    rapid open/close cycle. stop() must return promptly WITH the take's audio
    even when the underlying close blocks forever."""
    import time

    class WedgedStream:
        def abort(self):
            time.sleep(3600)  # PortAudio never comes back

        def close(self):
            pass

    r = localflow.Recorder()
    r.stream = WedgedStream()
    r.frames = [np.zeros((1600, 1), dtype=np.float32)] * 10  # 1s of audio
    t0 = time.time()
    audio = r.stop()
    took = time.time() - t0
    assert took < 3.0, f"stop() blocked {took:.1f}s on a wedged close"
    assert len(audio) == 16000, "the take's audio was lost in the wedge"
    assert r.stream is None
    print(f"PASS: wedged close abandoned in {took:.1f}s, audio kept")


def test_wedged_open_times_out_instead_of_eating_the_thread():
    """Pa_OpenStream can wedge too — start() must raise within its deadline,
    not strand the hotkey thread inside CoreAudio."""
    import time

    class WedgedInputStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            time.sleep(3600)

    real = localflow.sd.InputStream
    localflow.sd.InputStream = WedgedInputStream
    try:
        r = localflow.Recorder()
        t0 = time.time()
        try:
            r.start(timeout=0.5)
            raise AssertionError("start() returned despite a wedged open")
        except RuntimeError as e:
            assert "timed out" in str(e)
        took = time.time() - t0
        assert took < 2.0, f"start() blocked {took:.1f}s"
        assert r.stream is None
    finally:
        localflow.sd.InputStream = real
    print(f"PASS: wedged open raised in {took:.1f}s, thread survived")


def test_dead_stream_is_detected_and_audio_reset():
    """A long hold that produces no audio must NOT vanish silently: it must be
    called out and trigger an audio-engine reset so the NEXT take works
    without restarting the app."""
    lf = bare_app()
    lf.recorder = DeadStream(held_seconds=15.0)
    lf.recording = True

    resets = []
    localflow.reset_audio = lambda: resets.append(1) or True

    lf.on_release(HK)
    assert lf.recording is False
    assert lf.jobs.empty(), "a dead take must not reach the transcriber"
    assert resets, ("a 15s hold with zero audio must reset the audio engine — "
                    "without it the mic stays dead until an app restart")
    print("PASS: dead stream detected, audio engine reset")


def test_failed_open_resets_audio_for_the_next_press():
    """The -9986 open failure is the same wedge — after it, the engine must be
    reset so the SECOND press can succeed rather than failing forever."""
    lf = bare_app()
    lf.recorder = BoomOnStart()

    resets = []
    localflow.reset_audio = lambda: resets.append(1) or True

    lf.on_press(HK)
    assert lf.recording is False
    assert resets, "a failed mic open must reset the audio engine"
    print("PASS: failed open triggers an audio engine reset")


def test_worker_survives_a_paste_crash():
    """paste_text runs on the worker thread with no guard — if it ever raises,
    the worker dies and EVERY later dictation silently vanishes. One bad paste
    must cost one take, not the session."""
    import time

    import lexicon

    lexicon.log_dictation = lambda text: None

    lf = bare_app()
    lf.recording = False
    lf.transcribe = lambda audio: "hello there"
    lf.rules = []
    lf.vocab_words = set()

    calls = []

    def paste(text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("clipboard hiccup")

    localflow.paste_text = paste
    threading.Thread(target=lf.worker, daemon=True).start()
    lf.jobs.put((np.zeros(localflow.SAMPLE_RATE * 2, dtype=np.float32),
                 2.0, time.time()))
    lf.jobs.put((np.zeros(localflow.SAMPLE_RATE * 2, dtype=np.float32),
                 2.0, time.time()))
    deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert len(calls) == 2, ("the worker died on a paste error — every "
                             "dictation after it would vanish silently")
    print("PASS: worker survived a paste crash")


def test_paste_deferred_while_next_take_is_held():
    """Reproduce the mid-hold cutoff: pasting while the NEXT take is being
    recorded posts a Cmd-V whose forced flags overwrite the held modifier in
    the session state — mic_guard then reads the key as up and kills the take.
    The worker must therefore never call paste_text while recording is True."""
    import time

    import lexicon

    lexicon.log_dictation = lambda text: None  # don't pollute dictations.log

    lf = bare_app()
    lf.transcribe = lambda audio: "hello there"
    lf.rules = []
    lf.vocab_words = set()

    pasted = []
    localflow.paste_text = lambda text: pasted.append(
        (text, lf.recording))  # record whether a take was open at paste time

    lf.recording = True  # take #2 is being held while take #1's job runs
    threading.Thread(target=lf.worker, daemon=True).start()
    lf.jobs.put((np.zeros(localflow.SAMPLE_RATE * 2, dtype=np.float32),
                 2.0, time.time()))
    time.sleep(1.0)          # long enough to transcribe + (wrongly) paste
    lf.recording = False     # the held key is released
    deadline = time.time() + 3
    while not pasted and time.time() < deadline:
        time.sleep(0.05)
    assert pasted, "the deferred paste never happened after the take closed"
    text, was_recording = pasted[0]
    assert was_recording is False, (
        "paste_text ran while a take was open — this is what poisons the "
        "modifier state and cuts the take off mid-hold")
    print("PASS: paste deferred until the open take closed")


if __name__ == "__main__":
    try:
        test_failed_start_does_not_stick_or_propagate()
        test_failed_stop_does_not_stick_or_propagate()
        test_wedged_close_never_blocks_and_keeps_the_audio()
        test_wedged_open_times_out_instead_of_eating_the_thread()
        test_dead_stream_is_detected_and_audio_reset()
        test_failed_open_resets_audio_for_the_next_press()
        test_worker_survives_a_paste_crash()
        test_paste_deferred_while_next_take_is_held()
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
    print("ALL PASS")
