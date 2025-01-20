from groq import Groq

# Initialize Groq client with the API key
client = Groq(api_key="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")

def get_paraphrase(question):
    """
    Function to send a question to the Groq API and get the response.
    
    Args:
        question (str): The question to ask the Groq model.
    
    Returns:
        str: The model's response.
    """

    prompt = f"Paraphrase this query: {question}. Don't give any reasoning only the paraphrased version of the query. Give me 5 different versions."

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    res = ""
    for chunk in completion:
        res += chunk.choices[0].delta.content or ""
    
    return res
