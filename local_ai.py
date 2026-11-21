import ollama


def local_models():
    """Chat-capable models installed in Ollama, or [] if it isn't running."""
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return []
    return [n for n in names if "base" not in n and "embed" not in n]


def stream_local(prompt, model):
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.2},
        )
        for chunk in stream:
            text = chunk.message.content
            if text:
                yield text
    except ollama.ResponseError as e:
        if e.status_code == 404:
            yield f"AI request failed: model '{model}' is not installed. Run: ollama pull {model}"
        else:
            yield f"AI request failed: {e.error}"
    except Exception as e:
        yield f"AI request failed: could not reach Ollama ({e}). Is it running?"