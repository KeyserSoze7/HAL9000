"""
llm.py — llama.cpp / Qwen interface via llama-cpp-python
Handles prompt formatting, tool-call detection, and response generation.
"""

from llama_cpp import Llama
from tools import dispatch_tool, TOOL_DESCRIPTIONS

# --- Config ---
MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"

CONTEXT_LENGTH = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

SYSTEM_PROMPT = f"""You are a concise offline voice assistant. 
Keep responses short and conversational — 1-3 sentences max.
You have access to the following tools:

{TOOL_DESCRIPTIONS}

If the user's request matches a tool, respond ONLY with:
TOOL: <tool_name>
ARGS: <arg1>, <arg2>, ...

Otherwise, just answer naturally."""


def load_llm(model_path: str = MODEL_PATH) -> Llama:
    """Load the GGUF model via llama-cpp-python."""
    print(f"[LLM] Loading model from {model_path} ...")
    llm = Llama(
        model_path=model_path,
        n_ctx=CONTEXT_LENGTH,
        n_threads=4,            
        n_gpu_layers=0,         # pure CPU
        verbose=False,
    )
    print("[LLM] Model loaded.")
    return llm


def build_prompt(user_text: str, history: list[dict]) -> list[dict]:
    """Build the messages list for llama-cpp chat completion."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def generate(
    llm: Llama,
    user_text: str,
    history: list[dict],
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    Generate a response.
    Returns (response_text, updated_history).
    If a tool call is detected, executes it and returns the tool result as the response.
    """
    messages = build_prompt(user_text, history)

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    raw_response = output["choices"][0]["message"]["content"].strip()

    if verbose:
        print(f"[LLM] Raw response: {raw_response}")

    # --- Tool call detection ---
    if raw_response.startswith("TOOL:"):
        response_text = _handle_tool_call(raw_response, verbose)
    else:
        response_text = raw_response

    # Update history (keep last 6 turns to stay within context)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response_text})
    if len(history) > 12:
        history = history[-12:]

    return response_text, history


def _handle_tool_call(raw: str, verbose: bool) -> str:
    """Parse TOOL/ARGS lines and dispatch to tools.py."""
    try:
        lines = raw.strip().splitlines()
        tool_name = lines[0].replace("TOOL:", "").strip()
        args = []
        if len(lines) > 1:
            args_line = lines[1].replace("ARGS:", "").strip()
            args = [a.strip() for a in args_line.split(",") if a.strip()]

        if verbose:
            print(f"[LLM] Tool call detected: {tool_name}({args})")

        result = dispatch_tool(tool_name, args)
        return result

    except Exception as e:
        return f"I tried to use a tool but something went wrong: {e}"


# --- Quick standalone test ---
if __name__ == "__main__":
    llm = load_llm()
    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        response, history = generate(llm, user_input, history)
        print(f"Assistant: {response}\n")
