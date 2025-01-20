from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from groq import Groq

client = Groq(api_key="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")

def get_response():
    class Response(BaseModel):
        name: str = Field(description="The name of the person")
        age: str = Field(description="The age of the person")
    
    parser = JsonOutputParser(pydantic_object=Response)

    print(parser.get_format_instructions())

get_response()