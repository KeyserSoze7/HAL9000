"""
tools.py — Hardcoded function-calling tools
Add new tools by defining a function and registering it in TOOLS dict.
"""

import datetime
import subprocess
import platform
import os

# ─────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────

def get_time(args: list) -> str:
    """Return the current time."""
    now = datetime.datetime.now()
    return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}."


def get_weather(args: list) -> str:
    """
    Placeholder — fully offline so no live weather.
    Could be wired to an open-meteo API call if you ever go online.
    """
    return "I'm fully offline so I can't fetch live weather. You could wire me up to Open-Meteo for that."


def open_browser(args: list) -> str:
    """Open the default web browser."""
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", "https://www.google.com"])
        elif platform.system() == "Windows":
            os.startfile("https://www.google.com")
        else:
            subprocess.Popen(["xdg-open", "https://www.google.com"])
        return "Opening your browser."
    except Exception as e:
        return f"Couldn't open browser: {e}"


def list_files(args: list) -> str:
    """List files in the current directory or a given path."""
    path = args[0] if args else "."
    try:
        files = os.listdir(path)
        if not files:
            return f"No files found in {path}."
        return f"Files in {path}: " + ", ".join(files[:10])  # cap at 10
    except Exception as e:
        return f"Couldn't list files: {e}"


def system_info(args: list) -> str:
    """Return basic system information."""
    import psutil
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        ram_used = ram.used // (1024 ** 2)
        ram_total = ram.total // (1024 ** 2)
        return f"CPU at {cpu}%, RAM using {ram_used}MB of {ram_total}MB."
    except ImportError:
        return f"Running {platform.system()} {platform.release()}. Install psutil for more details."


def tell_joke(args: list) -> str:
    """Tell a hardcoded joke — because why not."""
    import random
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
        "Why was the JavaScript developer sad? Because they didn't know how to null their feelings.",
        "A SQL query walks into a bar, walks up to two tables and asks: Can I join you?",
    ]
    return random.choice(jokes)


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────

TOOLS: dict[str, callable] = {
    "get_time": get_time,
    "get_weather": get_weather,
    "open_browser": open_browser,
    "list_files": list_files,
    "system_info": system_info,
    "tell_joke": tell_joke,
}

# Shown to the LLM in its system prompt
TOOL_DESCRIPTIONS = """
- get_time: Returns the current time and date. Args: none.
- get_weather: Returns weather info (offline placeholder). Args: none.
- open_browser: Opens the default web browser. Args: none.
- list_files [path]: Lists files in a directory. Args: optional path (default current dir).
- system_info: Returns CPU and RAM usage. Args: none.
- tell_joke: Tells a programming joke. Args: none.
""".strip()


def dispatch_tool(tool_name: str, args: list) -> str:
    """Look up and call the requested tool. Returns result as a string."""
    tool_fn = TOOLS.get(tool_name.lower().strip())
    if tool_fn is None:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool '{tool_name}'. Available: {available}."
    return tool_fn(args)
