import os
from dotenv import load_dotenv
from huggingface_hub import login

def login_to_hf() -> None:
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

    if not HUGGING_FACE_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables.")

    login(token=HUGGING_FACE_TOKEN)