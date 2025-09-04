# app/services/llm_groq.py
import os
from groq import Groq

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client

def complete(messages, model=None, **kwargs) -> str:
    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    resp = _get_client().chat.completions.create(
        model=model, messages=messages,
        temperature=kwargs.get("temperature", 0.2),
        max_tokens=kwargs.get("max_tokens", 600),
        top_p=kwargs.get("top_p", 0.9),
        stream=False,
    )
    return resp.choices[0].message.content or ""

def complete_stream(messages, model=None, **kwargs):
    from groq import Groq
    import os
    _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    stream = _client.chat.completions.create(
        model=model, messages=messages,
        temperature=kwargs.get("temperature", 0.2),
        max_tokens=kwargs.get("max_tokens", 600),
        top_p=kwargs.get("top_p", 0.9),
        stream=True,
    )
    for chunk in stream:
        delta = None
        if chunk.choices and hasattr(chunk.choices[0], "delta"):
            delta = chunk.choices[0].delta.content
        elif chunk.choices and hasattr(chunk.choices[0], "message"):
            delta = getattr(chunk.choices[0].message, "content", None)
        if delta:
            yield delta
