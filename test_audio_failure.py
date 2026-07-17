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
    lf.toggle_take = False
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


class GoodRecorder:
    """A healthy take: 2 s of audio loud enough to pass the speech gate."""

    def __init__(self):
        import time
        self.frames = [1]          # non-empty: the callback "fired"
        self.t0 = time.time()
        self.device_name = "test mic"

    def start(self, timeout=1.5):
        import time
        self.t0 = time.time()

    def stop(self):
        return np.full(2 * localflow.SAMPLE_RATE, 0.1, dtype=np.float32)

    def silent_tail(self, *_):
        return False


def test_tap_arms_toggle_and_second_click_stops():
    """THE CLICK (Jul 18): a quick tap used to open the mic and silently drop
    the take as a graze — clicking looked completely broken. A tap must now
    ARM the take (click-to-toggle) and the next click must stop and queue it."""
    lf = bare_app()
    lf.recorder = GoodRecorder()
    lf.on_press(HK)                       # click down
    assert lf.recording, "press did not start a take"
    lf.on_release(HK)                     # click up, well inside TOGGLE_TAP
    assert lf.recording, "a quick tap must arm a toggle take, not drop it"
    assert lf.toggle_take, "toggle flag not armed by the tap"
    lf.on_press(HK)                       # second click ends the take
    assert lf.recording is False, "second click did not stop the take"
    assert not lf.jobs.empty(), "the toggle take never reached the transcriber"
    print("PASS: tap arms toggle, second click stops and queues the take")


def test_hold_release_still_finishes():
    """Hold-to-dictate must keep working exactly as before."""
    import time

    lf = bare_app()
    lf.recorder = GoodRecorder()
    lf.on_press(HK)
    lf.recorder.t0 = time.time() - 2      # the key was held for 2 s
    lf.on_release(HK)
    assert lf.recording is False
    assert not lf.jobs.empty(), "a held take never reached the transcriber"
    print("PASS: hold-release still finishes and queues the take")


class WrongMicRecorder:
    """THE DEVICE ROULETTE (Jul 17 23:40): the take records fine — full
    duration, frames arriving — but from a mic that can't hear you (an iPhone
    continuity mic on the desk), so the audio is voiceless end to end."""

    def __init__(self):
        import time
        self.t0 = time.time() - 5
        self.frames = [1]
        self.device_name = "Vin Microphone"

    def stop(self):
        return np.zeros(5 * localflow.SAMPLE_RATE, dtype=np.float32)


def test_voiceless_long_take_resets_audio():
    """A long take with no voice in it must reset the audio engine (fresh
    device scan) — that is the ONLY way the app ever escapes recording from
    the wrong mic without a restart. Every existing detector keys on missing
    frames and stays blind to this."""
    lf = bare_app()
    lf.recorder = WrongMicRecorder()
    lf.recording = True

    resets = []
    localflow.reset_audio = lambda: resets.append(1) or True

    lf.on_release(HK)
    assert lf.recording is False
    assert lf.jobs.empty(), "a voiceless take must not reach the transcriber"
    assert resets, ("a long voiceless take must reset the audio engine — "
                    "without it every take keeps coming from the wrong mic "
                    "until an app restart")
    print("PASS: voiceless long take triggers an audio engine reset")


def test_input_device_pinned_by_name():
    """The mic must be picked by NAME, not by 'system default' — the default
    is a moving target that continuity devices steal."""
    fake = [
        {"name": "Vin Microphone", "max_input_channels": 1},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    real = localflow.sd.query_devices
    localflow.sd.query_devices = lambda *a, **k: fake
    old = localflow.INPUT_DEVICE
    localflow.INPUT_DEVICE = "MacBook Pro Microphone"
    try:
        idx, name = localflow._resolve_input()
        assert idx == 1 and "MacBook" in name, (
            f"expected the built-in mic, got {idx} {name!r}")
    finally:
        localflow.sd.query_devices = real
        localflow.INPUT_DEVICE = old
    print("PASS: input device resolved by name, continuity mic skipped")


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
        test_tap_arms_toggle_and_second_click_stops()
        test_hold_release_still_finishes()
        test_voiceless_long_take_resets_audio()
        test_input_device_pinned_by_name()
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
    print("ALL PASS")
