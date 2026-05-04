import os

from openai import OpenAI


def make_client() -> OpenAI:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return OpenAI(base_url=base_url, api_key="ollama")


def default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "creative-agent")


def chat_extra() -> dict:
    """Extra kwargs every chat call must carry. Ollama defaults num_ctx to 2048,
    which silently truncates anything longer than that to the tail of the input -
    fatal for retriever/writer/critic, all of which send several thousand tokens."""
    return {
        "extra_body": {
            "options": {
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384")),
            }
        }
    }

def hello_world():
    client = make_client()
    model = default_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, world!"}],
        **chat_extra(),
    )
    print(resp.choices[0].message.content)
    client.close()
