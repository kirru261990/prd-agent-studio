from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from pathlib import Path
import fitz
import re

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

# RBI document registry
RBI_DOCS = {
    "Credit Card FAQs": "data/rbi/MD - Credit card FAQs.pdf",
    "Credit Cards Master Direction": "data/rbi/MD - Credit cards.pdf",
    "BBPS Guidelines": "data/rbi/MD - BBPS.pdf"
}

def extract_rbi_context(feature_description: str) -> str:
    """Extract relevant sections from RBI PDFs based on feature keywords."""
    keywords = [w for w in feature_description.lower().split() 
                if len(w) > 3 and w not in 
                ['this', 'that', 'with', 'from', 'they', 'their', 'will', 'have', 
                 'been', 'into', 'than', 'then', 'when', 'what', 'hdfc', 'pixel']]

    all_relevant = []

    for doc_name, doc_path in RBI_DOCS.items():
        if not Path(doc_path).exists():
            continue

        doc = fitz.open(doc_path)
        full_text = "\n".join([page.get_text() for page in doc])
        lines = full_text.split('\n')

        relevant_lines = []
        for i, line in enumerate(lines):
            if len(line.strip()) < 5:
                continue
            if any(kw in line.lower() for kw in keywords):
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                chunk = "\n".join(lines[start:end])
                if chunk not in relevant_lines:
                    relevant_lines.append(chunk)

        if relevant_lines:
            all_relevant.append(
                f"--- {doc_name} ---\n" + "\n\n".join(relevant_lines[:8])
            )

    if all_relevant:
        return "\n\n".join(all_relevant)
    else:
        return "No specific RBI section found for this feature. Proceed with general compliance awareness."


def load_pixel_context() -> str:
    """Load PIXEL Studio private context."""
    context_path = Path("pixel-context-private.md")
    return context_path.read_text() if context_path.exists() else ""


@app.get("/health")
def health():
    return {"status": "running", "agent": "prd-agent-studio"}


@app.post("/competitor-agent")
def competitor_agent(input: PRDInput):

    # Load PIXEL context
    pixel_context = load_pixel_context()

    # Extract feature short name for search
    feature_short = input.feature_description.split('\n')[0].replace('Feature: ', '')[:100]

    # Detect feature type
    is_rbi_feature = any(word in input.feature_description.lower()
                         for word in ['rbi', 'regulatory', 'compliance', 
                                      'guideline', 'circular', 'mandate', 
                                      'refund', 'bbps', 'autopay'])

    # Step 1a: Broad discovery search
    discovery_query = f"India fintech credit card {feature_short} 2024"
    discovery_results = tavily.search(
        query=discovery_query,
        max_results=3,
        search_depth="basic"
    )

    # Step 1b: Digital-first competitor targeted search
    competitor_query = f"OneCard Scapia Slice Kreditbee Freo {feature_short} India feature"
    competitor_results = tavily.search(
        query=competitor_query,
        max_results=5,
        search_depth="advanced"
    )

    # Step 1c: App store and community search for real user signal
    community_query = f"reddit OR \"google play\" {feature_short} credit card India user review 2024"
    community_results = tavily.search(
        query=community_query,
        max_results=3,
        search_depth="basic"
    )

    # Combine and deduplicate all competitor results
    all_results = (
        discovery_results.get('results', []) +
        competitor_results.get('results', []) +
        community_results.get('results', [])
    )
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r['url'] not in seen_urls:
            seen_urls.add(r['url'])
            unique_results.append(r)

    competitor_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in unique_results
    ])

    # Step 2: RBI context or global best practices
    if is_rbi_feature:
        secondary_context = extract_rbi_context(input.feature_description)
        secondary_label = "RBI REGULATORY CONTEXT"
    else:
        global_query = f"global best practice {feature_short} credit card fintech Monzo Revolut 2024"
        global_results = tavily.search(
            query=global_query,
            max_results=3,
            search_depth="basic"
        )
        secondary_context = "\n\n".join([
            f"Source: {r['url']}\n{r['content']}"
            for r in global_results.get('results', [])
        ])
        secondary_label = "GLOBAL BEST PRACTICES"

    # Step 3: Build prompt
    prompt = f"""A PM has asked you to research this feature:
"{input.feature_description}"

You have been given real search results and regulatory context below. 
Use ONLY this data to answer. Do not use training memory.
If something is not in the sources, explicitly say "not found in current sources."

COMPETITOR SEARCH RESULTS:
{competitor_context}

{secondary_label}:
{secondary_context}

Respond in exactly this structure:

COMPETITORS (max 3, must be specific digital-first Indian credit cards):
For each competitor found:
- Bank/Product name | Specific app or product name
- Why relevant: (1 sentence — why this competitor matters for this specific feature)
- What they built: (2 sentences max — be specific about how it works for the customer)
- One key point: (1 sentence — the single most differentiated thing about their implementation)
- Source: (URL)

WHITESPACE SIGNAL:
If fewer than 2 real competitors with this specific feature are found, state:
"FIRST-MOVER OPPORTUNITY: No significant competitor implementation found for this feature among digital-first Indian credit cards."
If competitors are found, skip this section.

WHAT REAL USERS SAY:
- 2-3 specific observations from Reddit, app store reviews, or community forums
- Include sentiment, specific complaints or praise, and source URL
- If no real user signal found, state "No community signal found in current sources"

{secondary_label} SUMMARY:
If RBI feature: Quote the most relevant RBI rule verbatim. Flag what PIXEL must comply with.
If non-RBI feature: 2-3 global examples with what works and why.

PIXEL STUDIO DIFFERENTIATION:
One paragraph only. Max 4 sentences.
Sentence 1: What PIXEL should build that no digital-first Indian competitor has done.
Sentence 2: Why this is defensible from first principles given PIXEL's technical architecture.
Sentence 3: How it stays RBI compliant (or leverages the RBI mandate as a feature, not a constraint).
Sentence 4: Why it does not add ops or CS overhead given PIXEL's current capabilities.

CONFIDENCE LEVEL: High / Medium / Low
Reason: (1 sentence explaining why)"""

    # Step 4: Call Claude with prompt caching
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": f"You are a senior competitive intelligence analyst for HDFC PIXEL Studio.\n\nHere is everything you need to know about PIXEL Studio — use this as your primary lens:\n\n{pixel_context}",
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
        "sources_searched": len(unique_results),
        "rbi_triggered": is_rbi_feature,
        "response": message.content[0].text
    }