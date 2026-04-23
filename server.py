from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

class PRDInput(BaseModel):
    feature_description: str

@app.get("/health")
def health():
    return {"status": "running", "agent": "prd-agent-studio"}

@app.post("/competitor-agent")
def competitor_agent(input: PRDInput):
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"You are a competitor research agent for a fintech credit card product team in India. Given this feature description: '{input.feature_description}', research and identify: 1) Top 3 competitors who have built something similar, 2) How they positioned it, 3) What PIXEL Studio's differentiation angle should be. Be specific and concise."
            }
        ]
    )
    return {"agent": "competitor", "response": message.content[0].text}
