from flask import Blueprint, jsonify


file_upload_routes = Blueprint("files",__name__)

@file_upload_routes.route("/uploads",methods=['POST'])
def file_upload():
    pass