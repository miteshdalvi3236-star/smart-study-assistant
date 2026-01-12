from flask import (
    Flask, render_template, request, send_file,
    redirect, url_for, jsonify, session, flash
)
import os
from io import BytesIO
import sqlite3
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ------------------------
# Summarizers
# ------------------------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

try:
    from transformers import pipeline
    hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    hf_summarizer = None
    print("HuggingFace model not loaded. Using LexRank fallback.")

# ------------------------
# Text-to-Speech
# ------------------------
from gtts import gTTS

# Optional PDF reader
try:
    import PyPDF2
except:
    PyPDF2 = None

# ------------------------
# App Config
# ------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
AUDIO_FOLDER = os.path.join("static", "audio")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------------
# Database
# ------------------------
DB_FILE = "history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # History table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            filename TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def add_history(action, filename, summary=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (action, filename, summary) VALUES (?, ?, ?)",
        (action, filename, summary)
    )
    conn.commit()
    conn.close()

init_db()

# ------------------------
# Auth Helpers
# ------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            flash("Please login first", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ------------------------
# Routes
# ------------------------

# ---- Home ----
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ---- Register ----
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password").strip()
        confirm_password = request.form.get("confirm_password").strip()

        if not (name and email and password and confirm_password):
            flash("Please fill all fields", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (email, hashed_password)
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "error")
            return redirect(url_for("register"))

    return render_template("register.html", title="Register")

# ---- Login ----
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("username").strip()
        password = request.form.get("password").strip()

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (email,))
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["user"] = email
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

    return render_template("login.html", title="Login")

# ---- Logout ----
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))

# ---- Dashboard ----
@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM history WHERE action='Upload & Summarize'")
    notes_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE action='Text Summarize'")
    summaries_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE action='Plan Created'")
    plans_count = c.fetchone()[0]
    conn.close()

    stats = {
        "notes": notes_count,
        "summaries": summaries_count,
        "plans": plans_count
    }

    return render_template("dashboard.html", stats=stats, title="Dashboard")

# ---- Other Pages ----
@app.route("/planner")
@login_required
def planner():
    return render_template("planner.html", title="Study Planner")

@app.route("/flashcards")
@login_required
def flashcards():
    return render_template("flashcards.html", title="Flashcards")

@app.route("/questions")
@login_required
def questions():
    return render_template("questions.html", title="Practice Questions")

# ---- History ----
@app.route("/history")
@login_required
def history_page():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, action, filename, summary, timestamp FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", history=rows, title="History")

@app.route("/history/<int:history_id>")
@login_required
def history_detail(history_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, action, filename, summary, timestamp FROM history WHERE id=?", (history_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "History not found", 404
    return render_template("history_detail.html", history=row, title="History Detail")

# ---- Upload & Summarize ----
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    message = ""
    original_text = ""
    summary_output = ""

    if request.method == "POST":
        file = request.files.get("notes_file")

        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if file.filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    original_text = f.read()
            elif file.filename.lower().endswith(".pdf") and PyPDF2:
                try:
                    with open(filepath, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                original_text += text + " "
                except Exception as e:
                    message = f"PDF read error: {e}"

            if original_text.strip():
                summary_output = generate_summary(original_text)
                add_history("Upload & Summarize", file.filename, summary_output)
                message = "File summarized successfully!"
            else:
                message = "No readable text found."
        else:
            message = "No file selected."

    return render_template("upload.html", message=message, original_text=original_text, summary=summary_output, title="Upload Notes")

# ---- Text Summarizer ----
@app.route("/summary", methods=["GET", "POST"])
@login_required
def summary_page():
    original_text = ""
    summary_output = ""
    message = ""

    if request.method == "POST":
        original_text = request.form.get("text", "").strip()
        if original_text:
            summary_output = generate_summary(original_text)
            add_history("Text Summarize", "Text Input", summary_output)
        else:
            message = "No text provided."

    return render_template("summary.html", original_text=original_text, summary=summary_output, message=message, title="Summarize Text")

# ---- Text to Speech ----
@app.route("/text-to-speech", methods=["POST"])
@login_required
def text_to_speech():
    text = request.form.get("text", "").strip()
    if len(text) < 5:
        return jsonify({"error": "Text too short"}), 400

    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    add_history("Voice Generated", filename, text[:150])

    return jsonify({"audio_url": f"/static/audio/{filename}"})

# ---- Download Summary ----
@app.route("/download_summary", methods=["POST"])
@login_required
def download_summary():
    summary_text = request.form.get("summary_text", "").strip()
    if not summary_text:
        return "No summary text", 400

    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, summary_text)

    buffer = BytesIO()
    buffer.write(pdf.output(dest="S").encode("latin-1"))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="summary.pdf", mimetype="application/pdf")

# ---- Summarization Logic ----
def generate_summary(text):
    text = text.strip()[:3000]

    if hf_summarizer:
        try:
            result = hf_summarizer(text, max_length=150, min_length=40, do_sample=False)
            return result[0]["summary_text"]
        except:
            pass

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lex = LexRankSummarizer()
    return " ".join(str(s) for s in lex(parser.document, 5))

# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
