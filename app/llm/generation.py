from app.utils.cache import get_cache, set_cache
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_key)

def generate(prompt: str):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + cheap
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # 🔥 important for RAG
        )

        answer =  response.choices[0].message.content.strip()
        # set_cache(prompt, answer)
        return answer
    except Exception as e:
        return f"Generation error: {str(e)}"
