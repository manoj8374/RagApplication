import json
from groq import Groq
import re

client = Groq(api_key="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")

def get_groq_response(question):
    insurance_prompt = (
        "You are a helpful assistant trained to answer questions based on the contents of an insurance PDF document. "
        "You are provided with the relevant parts of the insurance PDF that can help you answer the user’s question. "
        "Please respond with a valid JSON string that includes: "
        "{"
        "  \"answer\": \"The direct answer to the user's question.\","
        "  \"points\": ["
        "    {"
        "      \"title\": \"Point Title\","
        "      \"description\": \"Description of the point.\""
        "    },"
        "    {"
        "      \"title\": \"Another Point Title\","
        "      \"description\": \"Description of this point.\""
        "    }"
        "  ]"
        "}"
        "- If the answer includes multiple points or items, include them in an array under the 'points' key. Each point should be an object with a 'title' and 'description' key. "
        "- If the answer is simple and does not require multiple points, provide just the 'answer' key with a short description or direct answer. Don't include the 'points' key in the JSON."
        "If the user asks a question that is unrelated to the contents of the PDF, respond politely with: "
        "\"I’m sorry, but this question is unrelated to the insurance document. Please ask about the contents of the insurance policy.\""
    )

    full_question = insurance_prompt + "\n" + question

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": full_question}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        res = ""
        for chunk in completion:
            res += chunk.choices[0].delta.content or ""
        

        cleaned_res = re.sub(r'\\"', '"', res)  

        try:
            json_response = json.loads(cleaned_res)
            print("Parsed JSON Response:", json_response)
            return json_response
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", str(e))
            return {"error": "Failed to parse JSON", "response": cleaned_res}

    except Exception as e:
        return {"error": f"Error occurred: {str(e)}"}

