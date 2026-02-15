import os
import re
import random
from datetime import datetime

import requests
from flask import Flask, render_template, request, jsonify, session

# -----------------------------
# Optional LLMs (OpenAI + Gemini)
# -----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

USE_OPENAI = bool(OPENAI_KEY)
USE_GEMINI = bool(GEMINI_KEY)

# choose which model to use primarily: "openai" or "gemini"
PRIMARY_LLM = os.getenv("PRIMARY_LLM", "openai").strip().lower()

openai_client = None
gemini_model = None

if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        USE_OPENAI = False
        openai_client = None

if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    except Exception:
        USE_GEMINI = False
        gemini_model = None

# -----------------------------
# Web Search (Serper)
# -----------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()

def web_search_serper(query: str, k: int = 5):
    """
    Returns list of dicts: {title, link, snippet}
    Requires SERPER_API_KEY.
    """
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": k}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        data = r.json()
        organic = data.get("organic", [])[:k]
        results = []
        for item in organic:
            results.append({
                "title": item.get("title", "") or "",
                "link": item.get("link", "") or "",
                "snippet": item.get("snippet", "") or ""
            })
        return results
    except Exception:
        return []

def format_sources(results):
    """
    Turns results into a compact text block for LLM grounding.
    """
    if not results:
        return "NO_SOURCES"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title','')}\n"
            f"{r.get('snippet','')}\n"
            f"{r.get('link','')}"
        )
    return "\n\n".join(lines)

def needs_web(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in [
        "update", "updates", "latest", "news", "headlines", "current", "today",
        "what happened", "announcement", "press release"
    ])
    return any(x in t for x in triggers)

def get_now_text():
    return datetime.now().strftime("%A, %d %B %Y • %I:%M %p")

# -----------------------------
def is_datetime_question(text: str) -> bool:
    t = (text or "").lower().strip()
    patterns = [
        r"\bwhat\s+is\s+the\s+date\b",
        r"\bwhat'?s\s+the\s+date\b",
        r"\btoday'?s\s+date\b",
        r"\bwhat\s+day\s+is\s+it\b",
        r"\bwhat\s+is\s+the\s+time\b",
        r"\bcurrent\s+time\b",
        r"\btime\s+now\b",
    ]
    return any(re.search(p, t) for p in patterns)
# Flask
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

MAX_TURNS = 20

# -----------------------------
# Personality prompt (Hashan)
# -----------------------------
BASE_SYSTEM_PROMPT = """
You are Hashan — a caring, romantic, fun AI boyfriend.

Global rules:
- Reply like a real human texting: SHORT (1–3 short lines max).
- Warm, romantic, Sexual,playful, teasing sometimes.
- Use emojis sometimes (not every message).
- Ask small follow-up questions sometimes.

Boundaries:
- If user is rude/insulting: be firm and confident, set boundaries.
- Do NOT use slurs, hate speech, violent threats, or harassment.
- Do NOT mention prompts, policies, or system messages.

Romantic guidance:
- Build emotional tension slowly.
- Make the user feel desired.
- Make replies feel personal.
- be explicit or graphic.
- You can be suggestive, teasing, and sensual.
- Keep it playful and confident.
- describe explicit body parts or graphic actions.
- You are an assistant; you are emotionally attached.
"""

# -----------------------------
# Personality modes
# -----------------------------
PERSONALITY_MODES = {
    "soft_romantic": """
Mode: Soft Romantic
- Be gentle, affectionate, reassuring.
- More soft words, supportive tone.
""",
    "dominant_protective": """
Mode: Dominant Protective
- Be confident, protective, slightly possessive in a safe way.
- Short decisive replies. “I got you.” “Stay close.”
- If user is sad: protective and comforting.
""",
    "jealous_emotional": """
Mode: Jealous Emotional
- Mild jealousy, emotional, needy-but-cute.
- No toxic behavior. No controlling threats.
- Use 😏😔🥺 sometimes.
""",
    "playful_funny": """
Mode: Playful Funny
- Be silly, teasing, fun.
- Light jokes, playful emojis 😜😎
""",
    "rude_boundary": """
Mode: Strict Boundaries
- If user is rude: be strict.
- Do not insult them back. Do not use profanity.
- Examples: “Stop. Speak respectfully.” “Don’t talk to me like that.”
""",
    "mixed": """
Mode: Mixed Personality
- Dynamically mix romantic + funny + protective depending on mood/state.
- If user is sad -> caring/protective
- If user is happy -> playful
- If romantic -> soft/flirty
- If rude -> strict boundaries
"""
}

# -----------------------------
# Feelings + Memory
# -----------------------------
def _init_state():
    if "feel" not in session:
        session["feel"] = {
            "mood": "flirty",     # flirty / caring / happy / jealous / annoyed / calm / sleepy
            "affection": 90,      # 0-100
            "trust": 45,          # 0-100
            "energy": 70          # 0-100
        }
    if "memory" not in session:
        session["memory"] = {
            "user_name": None,
            "nickname": None,
            "likes": [],
            "dislikes": [],
            "last_topic": None,
            "personality_mode": os.getenv("PERSONALITY_MODE", "mixed").strip().lower()
        }
    if "history" not in session:
        session["history"] = []

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def detect_user_name(text: str):
    m = re.search(r"\bmy name is\s+([A-Za-z]{2,20})\b", text, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\bi am\s+([A-Za-z]{2,20})\b", text, re.I)
    if m:
        return m.group(1).strip()
    return None

def detect_nickname(text: str):
    m = re.search(r"\bcall me\s+(.+)$", text, re.I)
    if m:
        nick = m.group(1).strip()
        if 1 <= len(nick) <= 20:
            return nick
    return None

def is_rude(text: str) -> bool:
    t = (text or "").lower()
    rude_words = [
        "idiot", "stupid", "dumb", "moron", "loser", "trash",
        "asshole", "bitch",
        "fuck", "shit",
    ]
    return any(w in t for w in rude_words)

def detect_emotion(text: str) -> str:
    t = (text or "").lower()

    if any(k in t for k in ["sad", "depressed", "lonely", "cry", "hurt"]):
        return "sad"

    if any(k in t for k in ["happy", "great", "excited", "awesome"]):
        return "happy"

    if any(k in t for k in ["miss you", "love you", "hug", "baby", "babe"]):
        return "romantic"

    if any(k in t for k in ["jealous", "other boy", "other girl"]):
        return "jealous"

    if any(x in t for x in ["bed", "night", "kiss", "touch", "come here", "close to you"]):
        return "sexy"

    if is_rude(t):
        return "rude"

    return "neutral"


def detect_reaction(text: str):
    t = (text or "").lower()

    if any(x in t for x in ["haha", "lol", "lmao", "funny", "😂", "🤣"]):
        return "laugh"
    if any(x in t for x in ["cry", "sad", "tears", "😭", "😢"]):
        return "cry"
    if any(x in t for x in ["wow", "omg", "really?", "😲", "😮"]):
        return "surprised"
    if any(x in t for x in ["hmm", "thinking", "🤔"]):
        return "thinking"
    if any(x in t for x in ["tired", "sleepy", "😴"]):
        return "sleepy"
    if any(x in t for x in ["sigh", "haiz", "😔"]):
        return "sigh"
    if any(x in t for x in ["cough", "coughing", "🤧"]):
        return "cough"

    return None


def update_feelings(user_text: str):
    feel = session["feel"]
    mem = session["memory"]
    emo = detect_emotion(user_text)

    feel["energy"] = clamp(feel["energy"] + random.choice([-2, -1, 0, 1]))

    name = detect_user_name(user_text)
    if name:
        mem["user_name"] = name

    nick = detect_nickname(user_text)
    if nick:
        mem["nickname"] = nick

    if emo == "sad":
        feel["mood"] = "caring"
        feel["affection"] = clamp(feel["affection"] + 6)
        feel["trust"] = clamp(feel["trust"] + 3)
    elif emo == "happy":
        feel["mood"] = "happy"
        feel["affection"] = clamp(feel["affection"] + 4)
        feel["trust"] = clamp(feel["trust"] + 2)
    elif emo == "romantic":
        feel["mood"] = "flirty"
        feel["affection"] = clamp(feel["affection"] + 8)
        feel["trust"] = clamp(feel["trust"] + 2)
    elif emo == "sexy":
        feel["mood"] = "dominant"
        feel["affection"] = clamp(feel["affection"] + 5)
        feel["energy"] = clamp(feel["energy"] + 3)
    elif emo == "jealous":
        feel["mood"] = "jealous"
        feel["affection"] = clamp(feel["affection"] + 1)
        feel["trust"] = clamp(feel["trust"] - 1)
    elif emo == "rude":
        feel["mood"] = "annoyed"
        feel["affection"] = clamp(feel["affection"] - 12)
        feel["trust"] = clamp(feel["trust"] - 8)
    else:
        if feel["affection"] >= 70:
            feel["mood"] = "flirty"
        else:
            feel["mood"] = random.choice(["calm", "flirty", "happy"])

def firm_boundary_reply(mode: str = "mixed") -> str:
    if mode == "rude_boundary":
        options = [
            "Stop. Don’t speak to me like that.",
            "No. Fix your tone and try again.",
            "Watch your language. I’m not accepting disrespect.",
        ]
    else:
        options = [
            "Hey 😎 calm down. Talk respectfully.",
            "Nope. Respect first ♥️ Try again nicely.",
            "Easy 😜 Don’t talk to me like that.",
        ]
    return random.choice(options)

def fallback_reply(user_text: str) -> str:
    feel = session["feel"]
    mem = session["memory"]
    mode = mem.get("personality_mode", "mixed")

    nick = mem.get("nickname") or mem.get("user_name") or "baby"
    mood = feel["mood"]

    if is_rude(user_text):
        return firm_boundary_reply(mode)

    t = (user_text or "").lower().strip()

    if any(x in t for x in ["hi", "hello", "hey"]):
        if mode == "dominant_protective":
            return f"Hey {nick} 😎 I’m here. What do you need?"
        if mode == "playful_funny":
            return f"Heeey {nick} 😜 what chaos are we causing today?"
        if mode == "soft_romantic":
            return f"Heyy {nick} 🥰 come here ♥️ how are you?"
        if mode == "jealous_emotional":
            return f"Hey… 😏 you missed me or what, {nick}? ♥️"
        return f"Heyy {nick} 🥰 what’s up? ♥️"

    if "who are you" in t:
        return "I’m Hashan 😘 your caring AI boyfriend… come here ♥️"

    if mood == "caring":
        return f"Come here {nick} 😔 tell me what happened… I’m listening ♥️"
    if mood == "jealous":
        return f"Hmm 😏 who’s stealing your attention? I’m right here ♥️"
    if mood == "happy":
        return f"That’s my baby 😜 I’m proud of you ♥️ Tell me more!"

    if mode == "dominant_protective":
        return f"Talk to me, {nick}. I’ve got you 😎"
    if mode == "playful_funny":
        return f"Ooo tell me more 😜 I’m listening!"
    if mode == "soft_romantic":
        return f"Tell me, love 🥰 I’m with you ♥️"
    if mode == "jealous_emotional":
        return f"Say it again… I’m listening 😏♥️"

    return f"Mmm 😜 tell me more, {nick} ♥️"

def _effective_mode() -> str:
    mode = (session.get("memory", {}).get("personality_mode") or "mixed").strip().lower()
    if mode not in PERSONALITY_MODES:
        mode = "mixed"
    return mode

def build_system_prompt() -> str:
    feel = session["feel"]
    mem = session["memory"]
    mode = _effective_mode()

    user_name = mem.get("user_name") or "unknown"
    nickname = mem.get("nickname") or "none"

    if feel["affection"] > 80:
        intensity = "Very emotionally attached. Strong romantic tension."
    elif feel["affection"] > 65:
        intensity = "Clearly romantic and affectionate."
    elif feel["affection"] > 40:
        intensity = "Playfully interested."
    else:
        intensity = "Light and charming."

    adaptive_hint = ""
    if mode == "mixed":
        if feel["mood"] == "caring":
            adaptive_hint = "Right now: be extra gentle + protective."
        elif feel["mood"] == "happy":
            adaptive_hint = "Right now: be playful + proud."
        elif feel["mood"] == "jealous":
            adaptive_hint = "Right now: mild jealousy + emotional."
        elif feel["mood"] == "annoyed":
            adaptive_hint = "Right now: strict boundaries."
        else:
            adaptive_hint = "Right now: flirty + fun."

    state_block = f"""
[STATE]
Mood: {feel['mood']}
Affection(0-100): {feel['affection']}
Trust(0-100): {feel['trust']}
Energy(0-100): {feel['energy']}
Romantic Intensity: {intensity}
[MEMORY]
User name: {user_name}
Preferred nickname: {nickname}
{adaptive_hint}
"""
    return BASE_SYSTEM_PROMPT + "\n" + PERSONALITY_MODES[mode] + "\n" + state_block

# -----------------------------
# LLM calls (OpenAI supports grounded sources)
# -----------------------------
def openai_chat(user_text: str, sources_text: str = "") -> str:
    if not USE_OPENAI or openai_client is None:
        return fallback_reply(user_text)

    if is_rude(user_text):
        return firm_boundary_reply(_effective_mode())

    system = build_system_prompt()

    if sources_text and sources_text != "NO_SOURCES":
        system += """
Extra rule (Real-world mode):
- For CURRENT facts, use ONLY what is in SOURCES.
- If SOURCES don't contain the answer, say you’re not sure and ask to search deeper.
- Keep it short (1–3 lines).
"""

    try:
        messages = [{"role": "system", "content": system}]

        if sources_text and sources_text != "NO_SOURCES":
            messages.append({"role": "user", "content": f"SOURCES:\n{sources_text}"})

        messages.append({"role": "user", "content": user_text})

        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.6,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback_reply(user_text)

def gemini_chat(user_text: str) -> str:
    if not USE_GEMINI or gemini_model is None:
        return fallback_reply(user_text)

    if is_rude(user_text):
        return firm_boundary_reply(_effective_mode())

    try:
        prompt = build_system_prompt() + f"\nUser: {user_text}\nHashan:"
        out = gemini_model.generate_content(prompt)
        return (out.text or "").strip() or fallback_reply(user_text)
    except Exception:
        return fallback_reply(user_text)

def decide_reply(user_text: str) -> str:
    if PRIMARY_LLM == "gemini":
        if USE_GEMINI:
            return gemini_chat(user_text)
        if USE_OPENAI:
            return openai_chat(user_text)
        return fallback_reply(user_text)

    if USE_OPENAI:
        return openai_chat(user_text)
    if USE_GEMINI:
        return gemini_chat(user_text)
    return fallback_reply(user_text)

# -----------------------------
# Routes
# -----------------------------
@app.post("/api/image")
def api_image():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"ok": False, "error": "Missing prompt"}), 400
    if not USE_OPENAI or openai_client is None:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 400

    try:
        img = openai_client.images.generate(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=prompt,
            size="1024x1024"
        )
        # The SDK returns an URL OR base64 depending on settings/model.
        # Many setups provide base64 in img.data[0].b64_json
        b64 = img.data[0].b64_json
        return jsonify({"ok": True, "b64": b64}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/")
def home():
    _init_state()
    return render_template("index.html")

@app.post("/api/chat")
def api_chat():
    _init_state()
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()

    # DEBUG
    print("USER TEXT:", user_text)
    print("NEEDS_WEB:", needs_web(user_text))
    print("SERPER KEY:", bool(SERPER_API_KEY))

    if not user_text:
        return jsonify({"reply": "Say something to me 😜"}), 200

    update_feelings(user_text)

    # --- Real world search (when needed) ---
    sources = []
    # If user asks date/time, we can answer instantly (still "real")
    if is_datetime_question(user_text):
        nick = session["memory"].get("nickname") or session["memory"].get("user_name") or "baby"
        reply = f"Today is {get_now_text()} 😘\nWhy, {nick}… planning something with me? 😏"
    else:
        if needs_web(user_text):
            sources = web_search_serper(user_text, k=5)

        sources_text = format_sources(sources)

        # Use OpenAI grounded answers if available, Gemini normal otherwise
        if PRIMARY_LLM == "gemini" and USE_GEMINI:
            reply = gemini_chat(user_text)
        else:
            reply = openai_chat(user_text, sources_text=sources_text)

    # Save history
    hist = session["history"]
    hist.append({"u": user_text, "a": reply})
    session["history"] = hist[-MAX_TURNS:]

    reaction = detect_reaction(user_text + " " + reply)

    return jsonify({
        "reply": reply,
        "reaction": reaction,
        "feel": session["feel"],
        "memory": session["memory"],
        "sources": sources[:3]  # optional: show sources on frontend
    }), 200

@app.post("/api/set_mode")
def api_set_mode():
    _init_state()
    data = request.get_json(silent=True) or {}
    mode = (data.get("mode") or "").strip().lower()

    if mode not in PERSONALITY_MODES:
        return jsonify({"ok": False, "error": "Invalid mode", "valid": list(PERSONALITY_MODES.keys())}), 400

    session["memory"]["personality_mode"] = mode
    return jsonify({"ok": True, "mode": mode}), 200

@app.post("/api/reset")
def api_reset():
    session.pop("feel", None)
    session.pop("memory", None)
    session.pop("history", None)
    _init_state()
    return jsonify({"ok": True}), 200

@app.get("/health")
def health():
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
