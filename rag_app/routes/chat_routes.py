from flask import Blueprint, render_template, jsonify,request
from rag_app.services.llm_service import ask_llm
from rag_app.services.embed_service import get_embedding
import re

chat_route = Blueprint("chat", __name__)


@chat_route.route("/")
def index():
    return render_template("chat.html")


@chat_route.route("/ask", methods=["POST"])
def ask():
    request_payload = request.get_json()
    user_message = request_payload["message"]

    res = ask_llm(user_message)

    return jsonify({
        "status": "Success",
        "response": res
    })


@chat_route.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("document")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({    
            "status": "Error",  
            "message": "No file was selected."
        }), 400

    uploaded_file.seek(0, 2)
    file_size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)
    file_size_kb = round(file_size_bytes / 1024, 2)

    return jsonify({
        "status": "Success",
        "message": f"{uploaded_file.filename} uploaded successfully ({file_size_kb} KB)."
    })
