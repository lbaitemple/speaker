#!/usr/bin/env python3
"""
discord_app_speaker.py — Discord communication layer.

Listens to messages from configured Discord users and speaks them aloud
via Google Cloud TTS (using GoogleTTS from speechutil.py).

When STT_ENABLED=1, also captures microphone speech (using NoiseRobustSTT
from speechutil.py) and posts transcriptions to a Discord channel.

Environment variables:
- DISCORD_BOT_TOKEN     : bot token used for both listening and STT posting
- TARGET_APP_USER_ID   : comma-separated user IDs whose messages are spoken
- TARGET_CHANNEL_ID    : channel filter (0 = all visible channels)
- PROMPT_INITIATION    : prefix posted before each STT transcript (default: !new)
- DEFAULT_VOICE        : Google Cloud TTS voice (default: en-US-Chirp3-HD-Kore)
- TARGET_APP_USER_VOICES : USER_ID:VOICE,... per-user voice overrides
- STT_ENABLED          : set to 1 to enable microphone STT input
- STT_CHANNEL_ID       : channel to post STT transcripts (defaults to TARGET_CHANNEL_ID)
- LANGUAGE_CODE        : STT recognition language (default: en-US)
- STT_INPUT_GAIN       : software mic gain multiplier (default: 3.0)
- DEBUG_LOG            : set to 1 for verbose debug output
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from queue import Queue
from pathlib import Path

import discord
from dotenv import load_dotenv

from speechutil import GoogleTTS, NoiseRobustSTT, STT_AVAILABLE


# Fixed defaults for this workflow; env vars override these defaults.
FIXED_TARGET_CHANNEL_ID = 1515083840862031962
FIXED_TARGET_APP_USER_ID = 1505955976539541665
FIXED_PROMPT_INITIATION = "!new"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def env_int_set(name: str) -> set[int]:
    raw = os.getenv(name, "").strip()
    ids: set[int] = set()
    for item in raw.split(","):
        try:
            ids.add(int(item.strip()))
        except ValueError:
            pass
    return ids


def env_user_voices(name: str, default_voice: str) -> dict[int, str]:
    """Parse USER_ID:VOICE,... mappings from an env var."""
    raw = os.getenv(name, "").strip()
    mapping: dict[int, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        try:
            uid, voice = item.split(":", 1)
            mapping[int(uid.strip())] = voice.strip()
        except ValueError:
            pass
    return mapping


# ---------------------------------------------------------------------------
# Discord message helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"<@!?\d+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_message_text(message: discord.Message) -> str:
    if message.content and message.content.strip():
        return clean_text(message.content)

    parts: list[str] = []
    for emb in message.embeds:
        for piece in (emb.title, emb.description):
            if piece:
                parts.append(str(piece))
        for field in emb.fields:
            if field.name:
                parts.append(str(field.name))
            if field.value:
                parts.append(str(field.value))
    if parts:
        return clean_text(" ".join(parts))

    if message.attachments:
        return "Received a message with attachment."
    return ""


# ---------------------------------------------------------------------------
# STT → Discord thread
# ---------------------------------------------------------------------------

def stt_discord_loop(
    outbound_queue: Queue[tuple[int, str, str]],
    channel_id: int,
    lang_code: str,
    input_gain: float,
    prompt_initiation: str = FIXED_PROMPT_INITIATION,
) -> None:
    """
    Background thread: captures microphone speech with NoiseRobustSTT and
    queues each transcription for the connected Discord client to post.
    """
    if not STT_AVAILABLE:
        from speechutil import _stt_import_error
        print(f"STT disabled — missing dependency: {_stt_import_error}")
        print("Run: pip install pyaudio webrtcvad noisereduce numpy google-cloud-speech setuptools")
        return

    try:
        stt = NoiseRobustSTT.create(
            language_code=lang_code,
            input_gain=input_gain,
            vad_aggressiveness=0,
        )
    except Exception as e:
        logging.error(f"STT init failed: {e}")
        return

    stream = stt.open_stream()
    stt.calibrate_noise(stream)
    print("STT ready — speak to send messages to Discord.")

    while True:
        print("Listening... (speak now)", flush=True)
        transcript, _lang = stt.listen_once(stream)
        if not transcript:
            print("(no speech detected, listening again...)", flush=True)
            continue

        print(f"\nYou: {transcript}", flush=True)

        outbound_queue.put((channel_id, transcript, prompt_initiation))


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

class DiscordAppSpeaker(discord.Client):
    def __init__(
        self,
        target_app_user_ids: set[int],
        target_channel_id: int,
        tts: GoogleTTS,
        user_voices: dict[int, str],
        debug: bool,
        stt_enabled: bool = False,
        stt_channel_id: int = 0,
        lang_code: str = "en-US",
        input_gain: float = 3.0,
        prompt_initiation: str = FIXED_PROMPT_INITIATION,
    ):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.target_app_user_ids = target_app_user_ids
        self.target_channel_id = target_channel_id
        self.tts = tts
        self.user_voices = user_voices
        self.debug = debug
        self.stt_enabled = stt_enabled
        self.stt_channel_id = stt_channel_id
        self.lang_code = lang_code
        self.input_gain = input_gain
        self.prompt_initiation = prompt_initiation

        self._speech_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._stt_outbound_queue: Queue[tuple[int, str, str]] = Queue()
        self._speech_task: asyncio.Task | None = None
        self._stt_post_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _in_scope(self, message: discord.Message) -> bool:
        if self.target_channel_id == 0:
            return True
        if message.channel.id == self.target_channel_id:
            return True
        return getattr(message.channel, "parent_id", None) == self.target_channel_id

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        self._event_loop = asyncio.get_running_loop()

        print(f"Logged in as {self.user} ({self.user.id})")
        print(f"Watching user IDs: {sorted(self.target_app_user_ids)}")
        print(f"Default TTS voice: {self.tts.default_voice}")
        print(f"Channel scope: {self.target_channel_id or 'all visible'}")
        print(f"STT post channel: {self.stt_channel_id or self.target_channel_id}")
        print(f"STT prompt initiation: {self.prompt_initiation}")
        print(f"STT enabled: {self.stt_enabled}")

        await self._diagnose_channel_access()

        if self._speech_task is None or self._speech_task.done():
            self._speech_task = asyncio.create_task(self._speech_worker())

        if self._stt_post_task is None or self._stt_post_task.done():
            self._stt_post_task = asyncio.create_task(self._stt_post_worker())

        if self.stt_enabled:
            post_channel = self.stt_channel_id or self.target_channel_id
            if not post_channel:
                print("Warning: STT_ENABLED=1 but no channel configured — STT disabled")
            else:
                threading.Thread(
                    target=stt_discord_loop,
                    args=(self._stt_outbound_queue, post_channel, self.lang_code, self.input_gain, self.prompt_initiation),
                    daemon=True,
                ).start()
                print(f"STT input thread started — posting to channel {post_channel}")

    async def on_message(self, message: discord.Message) -> None:
        if self.user and message.author.id == self.user.id:
            return
        if message.author.id not in self.target_app_user_ids:
            return
        if not self._in_scope(message):
            return

        text = extract_message_text(message)
        if not text:
            self._log(f"Skipped empty message id={message.id}")
            return

        voice = self.user_voices.get(message.author.id, self.tts.default_voice)
        print(f"Bot: {text}\n", flush=True)
        await self._speech_queue.put((text, voice))

    # ------------------------------------------------------------------
    # TTS worker
    # ------------------------------------------------------------------

    async def _speech_worker(self) -> None:
        while True:
            text, voice = await self._speech_queue.get()
            try:
                await self.tts.speak(text, voice)
            except Exception as e:
                logging.error(f"TTS error: {e}")
            finally:
                self._speech_queue.task_done()

    async def _stt_post_worker(self) -> None:
        while True:
            channel_id, transcript, prompt_initiation = await asyncio.to_thread(self._stt_outbound_queue.get)
            try:
                initiation = (prompt_initiation or FIXED_PROMPT_INITIATION).strip()
                transcript_text = transcript.strip()

                if initiation:
                    await self.post_to_channel(initiation, channel_id)

                if transcript_text:
                    await self.post_to_channel(transcript_text, channel_id)
            except Exception as e:
                logging.error(f"Failed to post STT transcript to Discord: {e}")
            finally:
                self._stt_outbound_queue.task_done()

    async def _diagnose_channel_access(self) -> None:
        channel_id = self.stt_channel_id or self.target_channel_id
        if not channel_id or not self.user:
            return

        try:
            channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
        except discord.NotFound:
            print(
                f"Channel diagnostic: bot {self.user.id} cannot resolve channel {channel_id} "
                f"(Unknown Channel). Check server membership and channel visibility."
            )
            return
        except discord.Forbidden:
            print(
                f"Channel diagnostic: bot {self.user.id} is forbidden from accessing channel {channel_id}."
            )
            return
        except Exception as e:
            print(f"Channel diagnostic: failed to inspect channel {channel_id}: {e}")
            return

        channel_name = getattr(channel, "name", str(channel_id))
        guild = getattr(channel, "guild", None)
        guild_name = getattr(guild, "name", "DM/unknown")
        print(f"Channel diagnostic: resolved {channel_name} in {guild_name} ({channel_id})")

        permissions_for = getattr(channel, "permissions_for", None)
        guild_me = getattr(guild, "me", None)
        if callable(permissions_for) and guild_me is not None:
            perms = permissions_for(guild_me)
            print(
                "Channel diagnostic: permissions "
                f"view_channel={perms.view_channel} send_messages={perms.send_messages} "
                f"read_message_history={perms.read_message_history}"
            )

    # ------------------------------------------------------------------
    # Helpers called from threads
    # ------------------------------------------------------------------

    async def post_to_channel(self, text: str, channel_id: int) -> None:
        """Send a text message to a Discord channel (safe to call from any thread)."""
        try:
            channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
            await channel.send(text)
            self._log(f"Posted to channel {channel_id}: {text}")
        except Exception as e:
            logging.error(f"post_to_channel error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main() -> None:
    load_env()

    listener_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not listener_token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")

    target_app_user_ids = env_int_set("TARGET_APP_USER_ID") or {FIXED_TARGET_APP_USER_ID}
    target_channel_id = env_int("TARGET_CHANNEL_ID", FIXED_TARGET_CHANNEL_ID)
    default_voice = os.getenv("DEFAULT_VOICE", GoogleTTS.DEFAULT_VOICE).strip()
    user_voices = env_user_voices("TARGET_APP_USER_VOICES", default_voice)
    debug = env_flag("DEBUG_LOG", False)
    stt_enabled = env_flag("STT_ENABLED", False)
    stt_channel_id = env_int("STT_CHANNEL_ID", target_channel_id)
    lang_code = os.getenv("LANGUAGE_CODE", "en-US").strip()
    input_gain = float(os.getenv("STT_INPUT_GAIN", "3.0"))
    prompt_initiation = os.getenv("PROMPT_INITIATION", FIXED_PROMPT_INITIATION).strip() or FIXED_PROMPT_INITIATION

    tts = GoogleTTS(default_voice=default_voice)

    client = DiscordAppSpeaker(
        target_app_user_ids=target_app_user_ids,
        target_channel_id=target_channel_id,
        tts=tts,
        user_voices=user_voices,
        debug=debug,
        stt_enabled=stt_enabled,
        stt_channel_id=stt_channel_id,
        lang_code=lang_code,
        input_gain=input_gain,
        prompt_initiation=prompt_initiation,
    )

    try:
        await client.start(listener_token)
    finally:
        if not client.is_closed():
            await client.close()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    print("[startup] Setting PCM volume...")
    os.system("amixer -c 0 sset 'PCM' 100% >/dev/null 2>&1")
    try:
        print("[startup] Initializing Google TTS + Discord client...")
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Stopped.")
    except discord.LoginFailure:
        print("Discord login failed — check DISCORD_BOT_TOKEN in .env")


if __name__ == "__main__":
    main()
