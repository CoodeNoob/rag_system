from flask import Flask
from rag_app.routes.chat_routes import chat_route
from rag_app.routes.file_upload_routes import file_upload_routes

def start():
    app=Flask(__name__,template_folder="./templates")
    app.register_blueprint(chat_route)
    app.register_blueprint(file_upload_routes)

    return app
