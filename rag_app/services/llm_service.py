from dotenv import load_dotenv
import requests
import os

load_dotenv()

AI_URL = os.getenv("SERVER_URL")
MODEL_NAME = "aisingapore/llama-sea-lion-v3.5-8b-r:latest"

if AI_URL is None:
    raise ValueError("SERVER_URL is not set in .env file")


def prompt_build(question):
    return f"""
        သင်သည် မြန်မာဘာသာဖြင့် ဖြေကြားပေးသော AI ဖြစ်သည်။
        Question: 
        {question}
        မြန်မာဘာသာဖြင့်သာ ဖြေပါ။
        RULES:
        - Only Response to user questions
        - Do not include thinking 
        - DO NOT include <think> tags
        - DO NOT include reasoning
        - DO NOT include explanations about thinking
        - ONLY return the final answer
    """


def payload_prepare(question):
    return {
        "model" : MODEL_NAME,
        "prompt": prompt_build(question),
        "stream":False
    }


def ask_llm(question):
    response = requests.post(AI_URL, json=payload_prepare(question), timeout=30)
    response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        raise ValueError(f"LLM server did not return JSON. Status={response.status_code}, Body={response.text}")


