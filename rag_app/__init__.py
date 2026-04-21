from flask import Flask
from rag_app.routes.chat_routes import chat_route

def start():
    app=Flask(__name__,template_folder="./templates")
    app.register_blueprint(chat_route)
    return app
