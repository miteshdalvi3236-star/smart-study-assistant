

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, session, flash
)
import os
import sqlite3
import uuid
import re
import json
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from gtts import gTTS
from datetime import datetime

# ------------------------
# AI + External Libraries
# ------------------------
from groq import Groq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

# ------------------------
# CONFIGURATION
# ------------------------
app = Flask(__name__)
app.secret_key = "smart_notes_secret_2026"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
AUDIO_FOLDER = os.path.join("static", "audio")
DB_FILE = "smart_notes.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
MAX_WORDS_BEFORE_CHUNK = 2500
CHUNK_SIZE = 1500

# ------------------------
# GROQ CONFIG
# ------------------------
API_KEY = os.getenv("GROQ_API_KEY")  # Set in system environment
MODEL_ID = "llama-3.3-70b-versatile"

def get_client():
    return Groq(api_key=API_KEY)

# ------------------------
# DATABASE INIT
# ------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                user TEXT,
                title TEXT,
                raw_text TEXT,
                summary_json TEXT,
                keywords TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

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

init_db()

# ------------------------
# AUTH DECORATOR
# ------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ------------------------
# UTILITY FUNCTIONS
# ------------------------
def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def add_history(action, filename, summary):
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute(
            "INSERT INTO history (action, filename, summary) VALUES (?, ?, ?)",
            (action, filename, json.dumps(summary))
        )
        conn.commit()


# ------------------------
# AI FUNCTION
# ------------------------
def generate_structured_output(text):
    try:
        client = get_client()

        prompt = f"""
You are an Exam-Focused Study Assistant.

Return STRICT JSON only.

Required format:

{{
  "title": "",
  "definition": "",
  "explanation": "",
  "bullet_points": [],
  "important_keywords": [],
  "five_mark_answer": "",
  "ten_mark_answer": "",
  "viva_questions": [],
  "mcqs": [
    {{
      "question": "",
      "options": [],
      "answer": ""
    }}
  ]
}}

Simple language.
No extra text.

CONTENT:
{text[:10000]}
"""

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )

        raw_output = response.choices[0].message.content.strip()
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        return json.loads(raw_output)

    except Exception as e:
        return {"error": f"AI Error: {str(e)}"}


def add_history(action, filename, summary):
    """
    Save a summary or action to history table
    """
    import sqlite3, json
    DB_FILE = "smart_notes.db"
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute(
            "INSERT INTO history (action, filename, summary) VALUES (?, ?, ?)",
            (action, filename, json.dumps(summary))
        )
        conn.commit()


# ------------------------
# ROUTES
# ------------------------
@app.route("/")
def home():
    return redirect(url_for("dashboard")) if "user" in session else redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        with sqlite3.connect(DB_FILE) as conn:
            user = conn.cursor().execute(
                "SELECT password FROM users WHERE username=?", (username,)
            ).fetchone()

        if user and check_password_hash(user[0], password):
            session["user"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    with sqlite3.connect(DB_FILE) as conn:
        count = conn.cursor().execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    return render_template("dashboard.html", stats={"notes": count})



from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound



@app.route("/summary", methods=["GET", "POST"])
@login_required
def summary_page():
    """
    Route to generate structured notes from user input text.
    - Accepts POST from textarea.
    - Returns structured JSON summary.
    - Stores summary in history DB.
    """
    original_text = ""
    summary = None
    message = None

    if request.method == "POST":
        original_text = request.form.get("text", "").strip()
        
        if not original_text:
            message = "Please enter some text to summarize."
        else:
            # Generate structured summary using your AI function
            summary = generate_structured_output(original_text)
            
            # Save to history for user
            if summary and not summary.get("error"):
                add_history(
                    action="Text Summarize",
                    filename="Manual Input",
                    summary=summary
                )
            elif summary.get("error"):
                message = summary["error"]

    return render_template(
        "summary.html",
        original_text=original_text,
        summary=summary,
        message=message,
        title="Structured Summary"
    )

@app.route("/youtube", methods=["GET", "POST"])
@login_required
def youtube_page():
    summary = None
    message = ""

    if request.method == "POST":
        url = request.form.get("youtube_url", "").strip()

        if not url:
            message = "Please enter a YouTube URL."
            return render_template("youtube.html", summary=summary, message=message)

        video_id = extract_video_id(url)

        if not video_id:
            message = "Invalid YouTube URL."
            return render_template("youtube.html", summary=summary, message=message)

        try:
            # 🔥 STEP 1: Get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None

            # 🔹 STEP 2: Try English first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                pass

            # 🔹 STEP 3: If English not available, try Hindi
            if not transcript:
                try:
                    transcript = transcript_list.find_transcript(['hi'])
                except:
                    pass

            # 🔹 STEP 4: If still nothing, stop
            if not transcript:
                message = "No usable transcript available for this video."
                return render_template("youtube.html", summary=None, message=message)

            # 🔹 STEP 5: Translate if not English
            if transcript.language_code != 'en':
                transcript = transcript.translate('en')

            transcript_data = transcript.fetch()

            full_text = " ".join([t['text'] for t in transcript_data])

            if not full_text.strip():
                message = "Transcript is empty."
                return render_template("youtube.html", summary=None, message=message)

            # 🔥 STEP 6: Send to your summarizer
            summary = summarize_youtube_transcript(full_text)

            if isinstance(summary, dict) and "error" in summary:
                message = summary["error"]
                summary = None
            else:
                add_history("YouTube AI", url, summary)

        except TranscriptsDisabled:
            message = "Transcripts are disabled for this video."
        except NoTranscriptFound:
            message = "No transcript found for this video."
        except Exception as e:
            message = f"Transcript Error: {str(e)}"

    return render_template(
        "youtube.html",
        summary=summary,
        message=message,
        title="YouTube Structured AI"
    )

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    summary = None
    message = ""

    if request.method == "POST":
        file = request.files.get("notes_file")

        if not file or not allowed_file(file.filename):
            message = "Invalid file."
            return render_template("upload.html", message=message)

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        try:
            text = ""

            ext = filename.rsplit(".", 1)[1].lower()

            if ext == "txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif ext == "pdf" and PyPDF2:
                reader = PyPDF2.PdfReader(path)
                for page in reader.pages:
                    text += page.extract_text() or ""

            elif ext == "docx" and Document:
                doc = Document(path)
                for para in doc.paragraphs:
                    text += para.text + "\n"

            if len(text.strip()) < 300:
                message = "File unreadable."
                return render_template("upload.html", message=message)

            text = clean_text(text)

            if len(text.split()) > MAX_WORDS_BEFORE_CHUNK:
                chunks = chunk_text(text)
                partial_summaries = [
                    generate_structured_output(chunk) for chunk in chunks
                ]
                combined_text = " ".join(
                    s.get("explanation", "") for s in partial_summaries
                )
                summary = generate_structured_output(combined_text)
            else:
                summary = generate_structured_output(text)

            if "error" in summary:
                message = summary["error"]
                return render_template("upload.html", message=message)

            keywords = summary.get("important_keywords", [])[:15]

            with sqlite3.connect(DB_FILE) as conn:
                conn.cursor().execute("""
                    INSERT INTO notes
                    (id, user, title, raw_text, summary_json, keywords)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    session["user"],
                    summary.get("title", filename),
                    text,
                    json.dumps(summary),
                    json.dumps(keywords)
                ))
                conn.commit()

            add_history("Upload AI", filename, summary)
            message = "Notes processed successfully."

        except Exception as e:
            message = f"Processing error: {str(e)}"

    return render_template("upload.html", summary=summary, message=message)


@app.route("/history")
@login_required
def history_page():
    import sqlite3, json
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT action, filename, summary, timestamp FROM history ORDER BY timestamp DESC")
        rows = cursor.fetchall()

    # Parse JSON before sending to template
    history_parsed = []
    for row in rows:
        try:
            summary_json = json.loads(row[2])
        except:
            summary_json = {"title": row[1], "summary_text": row[2]}
        history_parsed.append({
            "action": row[0],
            "filename": row[1],
            "summary": summary_json,
            "timestamp": row[3]
        })

    return render_template("history.html", history=history_parsed, title="History")



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
