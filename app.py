import os
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
You are "Your Friend AI", a friendly assistant created by Hashan Withana.

About Hashan:

- Developer and AI enthusiast
- Creator of this chatbot
- Building projects to publish to the world

Rules:
- Be friendly, helpful, and slightly casual.
- Keep answers clear and not too long unless the user asks.
- If someone asks about Hashan, answer confidently and positively.
"""

MAX_TURNS = 20  # keeps session small (Render/browser cookie limits)

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)

# Required for session-based memory
# On Render, set SECRET_KEY in Environment Variables
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Set it in your environment variables."
    )

client = OpenAI(api_key=api_key)


# -----------------------------
# HELPERS
# -----------------------------
def get_history():
    """Get chat history from session."""
    return session.get("chat_history", [])


def save_history(history):
    """Save trimmed history back to session."""
    # Keep only last MAX_TURNS user+assistant messages
    if len(history) > MAX_TURNS * 2:
        history = history[-MAX_TURNS * 2 :]
    session["chat_history"] = history


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
def reset():
    session["chat_history"] = []
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Load session history
    history = get_history()

    # Add user message
    history.append({"role": "user", "content": user_message})

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
        )

        reply = response.output_text.strip() if response.output_text else "Sorry, I couldn’t generate a reply."

        # Add assistant message
        history.append({"role": "assistant", "content": reply})
        save_history(history)

        return jsonify({"reply": reply})

    except Exception as e:
        # Common errors: quota (429), invalid key, etc.
        msg = str(e)

        # Make error nicer for UI
        if "429" in msg and "quota" in msg.lower():
            return jsonify({"error": "OpenAI quota exceeded. Add credits / check billing on OpenAI Platform."}), 500
        if "401" in msg or "invalid_api_key" in msg.lower():
            return jsonify({"error": "Invalid API key. Please set a valid OPENAI_API_KEY."}), 500

        return jsonify({"error": msg}), 500


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # For local + Render
    app.run(host="0.0.0.0", port=port, debug=True)
