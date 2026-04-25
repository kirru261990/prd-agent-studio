from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PRDInput(BaseModel):
    feature_description: str

@app.get("/health")
def health():
    return {"status": "running", "agent": "prd-agent-studio"}

@app.post("/competitor-agent")
def competitor_agent(input: PRDInput):

    # Load PIXEL context
    context_path = Path("pixel-context-private.md")
    pixel_context = context_path.read_text() if context_path.exists() else ""

    # Step 1: Search for competitor information
    feature_short = input.feature_description.split('\n')[0].replace('Feature: ', '')[:100]
    search_query = f"digital credit card India {feature_short} OneCard Scapia Slice Uni 2024"
    search_results = tavily.search(
        query=search_query,
        max_results=5,
        search_depth="advanced"
    )

    # Step 2: Search for global best practices
    global_query = f"global best practice {feature_short} credit card fintech Klarna Apple Card Affirm 2024"
    global_results = tavily.search(
        query=global_query,
        max_results=3,
        search_depth="basic"
    )

    # Step 3: Format search results for Claude
    competitor_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in search_results.get('results', [])
    ])

    global_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in global_results.get('results', [])
    ])

    # Step 4: Build prompt
    prompt = f"""A PM has asked you to research this feature:
"{input.feature_description}"

You have been given real search results below. Use ONLY this data to answer. Do not use your training memory. If something is not in the search results, say "not found in current sources."

COMPETITOR SEARCH RESULTS:
{competitor_context}

GLOBAL BEST PRACTICE RESULTS:
{global_context}

Respond in exactly this structure:

COMPETITORS (max 3):
For each competitor:
- Bank name | Specific product/app name
- Why relevant: (1 sentence only)
- What they built: (2 sentences max — be specific, not generic)
- One key point: (1 sentence — the single most important thing about their implementation)
- Source: (URL)

GLOBAL BEST PRACTICES:
- 2-3 specific examples from global fintech leaders (Klarna, Apple Card, Affirm, Monzo, etc.)
- For each: what they do, why it works, source URL

PIXEL STUDIO DIFFERENTIATION:
One paragraph only. Max 4 sentences.
Sentence 1: What PIXEL should build that no competitor has done.
Sentence 2: Why this is defensible from first principles.
Sentence 3: How it stays RBI compliant.
Sentence 4: Why it does not add ops or CS overhead.

CONFIDENCE LEVEL: High / Medium / Low
(High = found specific data, Low = limited sources found)"""

    # Step 5: Call Claude with prompt caching
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": f"You are a senior competitive intelligence analyst for HDFC PIXEL Studio.\n\nHere is everything you need to know about PIXEL Studio:\n\n{pixel_context}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "agent": "competitor",
        "sources_searched": len(search_results.get('results', [])),
        "response": message.content[0].text
    }