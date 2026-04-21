from flask import Blueprint, render_template, jsonify,request
from rag_app.services.llm_service import ask_llm

chat_route = Blueprint("chat", __name__)


@chat_route.route("/")
def index():
    return render_template("chat.html")


@chat_route.route("/ask",methods=["POST"])
def ask():

    request_payload = request.get_json()

    user_message = request_payload["message"]

    res = ask_llm(user_message)


    print(f"THis is the response \n{res}")
    

    return jsonify({
        "message":"ok",
        "ok":True
    })
