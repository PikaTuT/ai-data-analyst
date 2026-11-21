import os, time
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv(usecwd=True))


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")          # local: from .env
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]     # cloud: from Secrets
    except Exception:
        return None


client = genai.Client(api_key=get_api_key())

MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]

FAST = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="LOW")
)


def ask_gemini(prompt, retries=2):
    """Non-streaming version: returns the full answer as one string."""
    start = time.time()
    for model in MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=FAST
                )
                secs = time.time() - start
                print(f"[gemini] {model} answered in {secs:.1f}s")
                return f"{response.text}\n\n_(answered by {model} in {secs:.1f}s)_"
            except Exception as e:
                msg = str(e)
                print(f"[gemini] {model} attempt {attempt + 1} failed at {time.time() - start:.1f}s: {msg[:80]}")
                if "503" in msg or "429" in msg:
                    time.sleep(1)
                elif "404" in msg:
                    break
                else:
                    return f"AI request failed: {e}"
    return "No model could answer right now. Please try again later."


def stream_gemini(prompt):
    """Streaming version: yields the answer piece by piece."""
    for model in MODELS:
        got_text = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=prompt, config=FAST
            ):
                if chunk.text:
                    got_text = True
                    yield chunk.text
            return
        except Exception as e:
            msg = str(e)
            if got_text:
                yield f"\n\n[Stream interrupted: {msg[:80]}]"
                return
            if not any(code in msg for code in ("503", "429", "404")):
                yield f"AI request failed: {e}"
                return
            print(f"[gemini] {model} unavailable, trying next: {msg[:60]}")
    yield "No model could answer right now. Please try again later."