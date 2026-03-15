import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def groq_response(prompt):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content

def generate_llm_response(prompt):
    try:
        return groq_response(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "AI analysis temporarily unavailable."