#!/usr/bin/env python3
"""
speechutil.py — reusable STT and TTS utilities.

Classes:
    NoiseRobustSTT  — microphone capture + WebRTC VAD + Google Speech-to-Text
    GoogleTTS       — Google Cloud Text-to-Speech with mpg123/ffplay playback
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile

from google.cloud import texttospeech

try:
    import numpy as np
    import pyaudio as _pyaudio
    import webrtcvad as _webrtcvad
    from google.cloud import speech as _speech
    STT_AVAILABLE = True
except ImportError as _e:
    STT_AVAILABLE = False
    _stt_import_error = str(_e)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class NoiseRobustSTT:
    """
    Noise-robust speech-to-text using WebRTC VAD + Google Cloud Speech.
    Ported from ai_app7.py with robot-specific display calls removed.

    Usage:
        stt = NoiseRobustSTT.create(language_code="en-US", input_gain=3.0)
        stream = stt.open_stream()
        stt.calibrate_noise(stream)
        while True:
            transcript, lang = stt.listen_once(stream)
            if transcript:
                print(transcript)
    """

    def __init__(
        self,
        speech_client,
        py_audio,
        sample_rate: int = 16000,
        chunk_size: int = 320,
        vad_aggressiveness: int = 0,
        language_code: str = "en-US",
        input_gain: float = 3.0,
    ):
        if not STT_AVAILABLE:
            raise RuntimeError(f"STT dependencies not available: {_stt_import_error}")

        self.speech_client = speech_client
        self.py_audio = py_audio
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.language_code = language_code
        self.input_gain = input_gain

        self.vad = _webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_duration_ms = 20

        self.num_silent_frames_threshold = 25   # ~0.5 s silence to end utterance
        self.num_speech_frames_threshold = 2    # ~40 ms to confirm speech start
        self.padding_frames = 10

        self.noise_profile = None
        self.calibration_time = 2.0
        self.noise_reduction_strength = 0.5
        self.use_stationary_noise = False
        self.silence_threshold = 500
        self.enable_noise_reduction = False

        self.accumulated_audio: list = []
        self.ring_buffer: list = []
        self.ring_buffer_size = 30
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False

        logging.info(
            f"NoiseRobustSTT ready (vad_aggressiveness={vad_aggressiveness}, "
            f"lang={language_code}, gain={input_gain}x)"
        )

    @classmethod
    def create(cls, language_code: str = "en-US", input_gain: float = 3.0, vad_aggressiveness: int = 0) -> "NoiseRobustSTT":
        """Convenience factory: initialises PyAudio and Speech client automatically."""
        if not STT_AVAILABLE:
            raise RuntimeError(f"STT dependencies not available: {_stt_import_error}")

        # Boost hardware capture volume
        for ctrl in ("Capture", "Mic", "Microphone", "ADC Capture Volume"):
            os.system(f"amixer -c 0 sset '{ctrl}' 100% cap 2>/dev/null")

        py_audio = _pyaudio.PyAudio()
        speech_client = _speech.SpeechClient()

        device_info = py_audio.get_default_input_device_info()
        native_rate = int(device_info["defaultSampleRate"])
        chunk_size = int(native_rate * 0.020)  # 20 ms frame

        return cls(
            speech_client=speech_client,
            py_audio=py_audio,
            sample_rate=native_rate,
            chunk_size=chunk_size,
            vad_aggressiveness=vad_aggressiveness,
            language_code=language_code,
            input_gain=input_gain,
        )

    def open_stream(self):
        """Open a PyAudio input stream at the device's native rate."""
        return self.py_audio.open(
            format=_pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_gain(self, audio_data):
        if self.input_gain == 1.0:
            return audio_data
        return np.clip(
            audio_data.astype(np.float32) * self.input_gain, -32768, 32767
        ).astype(np.int16)

    def _reduce_noise(self, audio_data):
        if not self.enable_noise_reduction or self.noise_profile is None:
            return audio_data
        try:
            import noisereduce as nr
            reduced = nr.reduce_noise(
                y=audio_data.astype(np.float32),
                sr=self.sample_rate,
                y_noise=self.noise_profile.astype(np.float32),
                stationary=self.use_stationary_noise,
                prop_decrease=self.noise_reduction_strength,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
                n_fft=2048,
                clip_noise_stationary=True,
            )
            return reduced.astype(np.int16)
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}")
            return audio_data

    _VAD_RATE = 16000

    def _resample_to_vad_rate(self, audio_data):
        if self.sample_rate == self._VAD_RATE:
            return audio_data
        from math import gcd
        g = gcd(int(self.sample_rate), self._VAD_RATE)
        up = self._VAD_RATE // g
        down = int(self.sample_rate) // g
        n_out = int(len(audio_data) * up / down)
        indices = (np.arange(n_out) * down / up).astype(int)
        return audio_data[np.clip(indices, 0, len(audio_data) - 1)]

    def _is_speech(self, audio_data) -> bool:
        try:
            resampled = self._resample_to_vad_rate(audio_data)
            expected = int(self._VAD_RATE * self.vad_frame_duration_ms / 1000)
            if len(resampled) < expected:
                resampled = np.pad(resampled, (0, expected - len(resampled)), "constant")
            else:
                resampled = resampled[:expected]
            return self.vad.is_speech(resampled.tobytes(), self._VAD_RATE)
        except Exception as e:
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            result = rms >= self.silence_threshold
            logging.warning(f"VAD fallback (RMS={rms:.0f}, speech={result}): {e}")
            return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_noise(self, stream) -> None:
        """Capture 2 s of ambient noise to set the silence threshold."""
        print("\n" + "=" * 60)
        print("Calibrating noise — please stay silent for 2 seconds...")
        print("=" * 60)

        noise_samples = []
        frames_needed = int(self.sample_rate / self.chunk_size * self.calibration_time)
        for i in range(frames_needed):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frame = self._apply_gain(np.frombuffer(data, dtype=np.int16))
            noise_samples.append(frame)
            if (i + 1) % 8 == 0:
                print(f"  {(i + 1) / frames_needed * 100:.0f}%...", end="\r", flush=True)

        self.noise_profile = np.concatenate(noise_samples)
        noise_rms = np.sqrt(np.mean(self.noise_profile.astype(np.float32) ** 2))
        self.silence_threshold = min(1000, max(200, noise_rms * 2.0))

        print(f"\nCalibration done. RMS={noise_rms:.0f}, threshold={self.silence_threshold:.0f}, gain={self.input_gain}x")
        print("=" * 60 + "\n")
        logging.info(f"Calibration done. RMS={noise_rms:.0f}, threshold={self.silence_threshold:.0f}")

    def listen_once(self, stream) -> tuple[str | None, str | None]:
        """
        Block until one complete utterance is captured.
        Returns (transcript, detected_language_code) or (None, None).
        """
        self.accumulated_audio = []
        self.ring_buffer = []
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False
        speech_detected = False

        max_iterations = int(30 * self.sample_rate / self.chunk_size)
        frames_per_print = max(1, int(0.2 * self.sample_rate / self.chunk_size))

        for iteration in range(max_iterations):
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = self._apply_gain(np.frombuffer(data, dtype=np.int16))
                cleaned = self._reduce_noise(audio_data)
                is_speech = self._is_speech(cleaned)

                # Live level meter
                if iteration % frames_per_print == 0:
                    rms = np.sqrt(np.mean(cleaned.astype(np.float32) ** 2))
                    bar = "█" * min(20, int(rms / 100)) + "░" * max(0, 20 - int(rms / 100))
                    status = "SPEECH" if is_speech else "quiet "
                    print(f"\r  [{bar}] RMS={rms:5.0f} {status}", end="", flush=True)

                if is_speech:
                    self.consecutive_speech_frames += 1
                    self.consecutive_silent_frames = 0

                    if (not self.is_currently_speaking
                            and self.consecutive_speech_frames >= self.num_speech_frames_threshold):
                        self.is_currently_speaking = True
                        if not speech_detected:
                            speech_detected = True
                            print(f"\n  [recording...]", flush=True)
                            if self.ring_buffer:
                                self.accumulated_audio.extend(self.ring_buffer)
                                self.ring_buffer = []

                    if self.is_currently_speaking:
                        self.accumulated_audio.append(cleaned)
                    else:
                        self._push_ring(cleaned)
                else:
                    self.consecutive_silent_frames += 1
                    self.consecutive_speech_frames = 0

                    if self.is_currently_speaking and self.consecutive_silent_frames <= self.padding_frames:
                        self.accumulated_audio.append(cleaned)
                    elif not self.is_currently_speaking:
                        self._push_ring(cleaned)

                if (self.is_currently_speaking
                        and self.consecutive_silent_frames >= self.num_silent_frames_threshold):
                    print("\n  [processing...]", flush=True)
                    if not self.accumulated_audio:
                        return None, None

                    full_audio = np.concatenate(self.accumulated_audio)
                    if len(full_audio) / self.sample_rate < 0.5:
                        logging.warning("Audio too short (<0.5 s), ignoring")
                        return None, None

                    return self._transcribe(full_audio.tobytes())

            except Exception as e:
                logging.error(f"listen_once error: {e}")
                return None, None

        print("\n  [timeout — no speech in 30 s]", flush=True)
        return None, None

    def _push_ring(self, frame):
        self.ring_buffer.append(frame)
        if len(self.ring_buffer) > self.ring_buffer_size:
            self.ring_buffer.pop(0)

    def _transcribe(self, audio_bytes: bytes) -> tuple[str | None, str | None]:
        """Send audio bytes to Google Speech-to-Text and return (transcript, lang)."""
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            duration = len(audio_array) / self.sample_rate
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            logging.info(f"Transcribing: {duration:.2f}s RMS={rms:.0f}")

            model = "default" if duration < 3.0 else "latest_long"
            config = _speech.RecognitionConfig(
                encoding=_speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model=model,
                use_enhanced=True,
                audio_channel_count=self.channels,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                alternative_language_codes=[
                    "zh-CN", "zh-TW", "es-ES", "fr-FR", "de-DE",
                    "ja-JP", "ko-KR", "pt-BR", "ru-RU", "it-IT", "he-IL",
                ],
            )
            response = self.speech_client.recognize(
                config=config,
                audio=_speech.RecognitionAudio(content=audio_bytes),
            )

            if response.results and response.results[0].alternatives:
                alt = response.results[0].alternatives[0]
                try:
                    detected_lang = response.results[0].language_code or self.language_code
                except Exception:
                    detected_lang = self.language_code
                logging.info(f"Transcript: '{alt.transcript}' ({alt.confidence:.0%}, lang={detected_lang})")
                return alt.transcript, detected_lang

            logging.warning("No transcript returned — audio may be too quiet or unclear")
            return None, None

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None, None


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class GoogleTTS:
    """
    Google Cloud Text-to-Speech with mpg123/ffplay playback.

    Usage (async):
        tts = GoogleTTS()
        await tts.speak("Hello!")

    Usage (sync, from a thread):
        tts = GoogleTTS()
        tts.speak_sync("Hello!")
    """

    DEFAULT_VOICE = "en-US-Chirp3-HD-Kore"
    _PLAYERS = [
        ["mpg123", "-q"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ]

    def __init__(self, default_voice: str = DEFAULT_VOICE):
        self.default_voice = default_voice
        self.client = texttospeech.TextToSpeechClient()

    def _synthesize(self, text: str, voice: str) -> str:
        """Synthesise speech and write to a temp MP3 file. Returns the file path."""
        lang_code = "-".join(voice.split("-")[:2])
        response = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            ),
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(response.audio_content)
        tmp.close()
        return tmp.name

    async def speak(self, text: str, voice: str | None = None) -> None:
        """Synthesise and play audio asynchronously (for use inside asyncio)."""
        voice = voice or self.default_voice
        mp3_path = await asyncio.get_event_loop().run_in_executor(
            None, self._synthesize, text, voice
        )
        try:
            for player in self._PLAYERS:
                try:
                    proc = await asyncio.create_subprocess_exec(*player, mp3_path)
                    await proc.communicate()
                    if proc.returncode == 0:
                        return
                except FileNotFoundError:
                    continue
            raise RuntimeError("No audio player found — install mpg123 or ffplay.")
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

    def speak_sync(self, text: str, voice: str | None = None) -> None:
        """Synthesise and play audio synchronously (for use in threads)."""
        voice = voice or self.default_voice
        mp3_path = self._synthesize(text, voice)
        try:
            for player in self._PLAYERS:
                try:
                    result = subprocess.run(player + [mp3_path])
                    if result.returncode == 0:
                        return
                except FileNotFoundError:
                    continue
            raise RuntimeError("No audio player found — install mpg123 or ffplay.")
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass
