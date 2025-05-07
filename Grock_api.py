from dotenv import load_dotenv
import os
from groq import Groq  # make sure this library is installed


load_dotenv()  # Automatically loads environment variables from .env


def chat(QUERY, system_prompt):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # ✅ correct model name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUERY}
        ],
        temperature=0.8,
        max_tokens=60,
        top_p=0.9,
        stream=False
    )
    #print(response.choices[0].message.content)  # ✅ correct way to access response
    return response.choices[0].message.content  # ✅ correct way to return response
