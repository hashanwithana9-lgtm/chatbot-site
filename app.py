import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Load API key safely from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Choose model
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": "You are Your Friend, a smart AI assistant created by Hashan Withana. Never say you are ChatGPT.introduce yourself as Your Friend."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        reply = response.output_text

        return jsonify({"reply": reply})

    except Exception as e:
        print("FULL ERROR:", e)
        return jsonify({"error": str(e)}), 500


# Important for local + Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
