from dotenv import load_dotenv
import requests
import os
from rag_app.data_sourcess.vector_storage import search

load_dotenv()

AI_URL = os.getenv("SERVER_URL")
MODEL_NAME = "gemma4:e2b"

if AI_URL is None:
    raise ValueError("SERVER_URL is not set in .env file")


def prompt_build(question, context):
    return f"""
သင်သည် AGGA.IO ရဲ့ AI Assistant ဖြစ်သည်။

Context:
{context}

Question:
{question}

RULES:
- Context ထဲမှသာ ဖြေပါ
- မတွေ့ပါက "မသိပါ" ဟုပြောပါ
- မြန်မာဘာသာဖြင့်သာ ဖြေပါ
- Only return final answer
- Do not include thinking or explanation
- ယဉ်ကျေးပြီး စိတ်ရှည်ပါ။
- You are a helpful assistant. Answer clearly with full details.
- 
"""


# 🚀 Main function
def ask_llm(question):
    docs = search(question, k=3)

    context = "\n".join(docs)

    full_prompt = prompt_build(question, context)

    response = requests.post(
        f"{AI_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "")