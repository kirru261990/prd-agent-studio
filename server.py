from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import json
import re
from pathlib import Path
from datetime import datetime
import fitz

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

app.mount("/ui", StaticFiles(directory=".", html=True), name="ui")

# RBI document registry
RBI_DOCS = {
    "Credit Card FAQs": "data/rbi/MD - Credit card FAQs.pdf",
    "Credit Cards Master Direction": "data/rbi/MD - Credit cards.pdf",
    "BBPS Guidelines": "data/rbi/MD - BBPS.pdf"
}

# Input models
class ClarifyInput(BaseModel):
    feature_name: str
    feature_description: str
    competitors: str = ""
    qa_history: list = []

class AnalysisInput(BaseModel):
    feature_name: str
    feature_description: str
    competitors: str = ""
    qa_history: list = []


def extract_rbi_context(feature_description: str) -> str:
    keywords = [w for w in feature_description.lower().split()
                if len(w) > 3 and w not in
                ['this', 'that', 'with', 'from', 'they', 'their', 'will', 'have',
                 'been', 'into', 'than', 'then', 'when', 'what', 'hdfc', 'pixel',
                 'play', 'user', 'users', 'card', 'credit', 'feature']]

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

    return "\n\n".join(all_relevant) if all_relevant else \
        "No specific RBI section found. Proceed with general compliance awareness."


def load_pixel_context() -> str:
    context_path = Path("pixel-context-private.md")
    return context_path.read_text() if context_path.exists() else ""


def save_analysis(feature_name: str, result: dict):
    output_dir = Path("data/analyses")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^a-z0-9]', '_', feature_name.lower())[:50]
    filename = output_dir / f"{timestamp}_{safe_name}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    return str(filename)


def format_qa_history(qa_history: list) -> str:
    if not qa_history:
        return ""
    lines = ["\nCLARIFYING QUESTIONS AND ANSWERS:"]
    for qa in qa_history:
        lines.append(f"Q: {qa['question']}")
        lines.append(f"A: {qa['answer']}")
    return "\n".join(lines)


@app.get("/health")
def health():
    return {"status": "running", "agent": "prd-agent-studio"}


@app.post("/clarify")
def clarify(input: ClarifyInput):
    """Generate next clarifying question or signal READY if enough context."""

    pixel_context = load_pixel_context()
    qa_so_far = format_qa_history(input.qa_history)
    question_number = len(input.qa_history) + 1

    prompt = f"""You are a senior product analyst. A PM has described a feature and you need to ask ONE clarifying question before running competitive research.

    Feature name: {input.feature_name}
    Feature description: {input.feature_description}
    {qa_so_far}

    Study the feature description carefully. Identify the single most important thing that is ambiguous, missing, or unclear that would most change the direction of the competitive research if you knew it.

    Do NOT ask about:
    - Things already clearly stated in the description
    - Generic questions like "who is the target user" if that's already obvious
    - Regulatory context if the feature already mentions RBI
    - Things that won't change the research direction

    DO ask about:
    - The primary business goal if it's ambiguous (cost saving vs revenue vs retention vs compliance)
    - Edge cases that would change the competitive set entirely
    - Constraints that would eliminate certain differentiation angles
    - The specific trigger or moment in the user journey if unclear

    If after {len(input.qa_history)} answers you have enough context for excellent research, respond with: READY

    Otherwise respond with ONE sharp, specific question. No preamble. No explanation. Just the question."""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=f"You are a senior product analyst for HDFC PIXEL Studio.\n\n{pixel_context}",
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()
    is_ready = response_text.upper() == "READY" or question_number > 3

    return {
        "question": response_text if not is_ready else None,
        "ready": is_ready,
        "question_number": question_number
    }


@app.post("/competitor-agent")
def competitor_agent(input: AnalysisInput):

    pixel_context = load_pixel_context()

    # Build enriched feature description from Q&A
    qa_context = format_qa_history(input.qa_history)
    enriched_description = f"{input.feature_description}\n{qa_context}"

    feature_short = input.feature_name[:100]

    # Detect RBI feature
    is_rbi_feature = any(word in enriched_description.lower()
                         for word in ['rbi', 'regulatory', 'compliance',
                                      'guideline', 'circular', 'mandate',
                                      'refund', 'bbps', 'autopay'])

    # Build competitor search list
    default_competitors = "OneCard Scapia Slice Kreditbee Freo Uni"
    custom_competitors = input.competitors.strip() if input.competitors.strip() else ""
    all_competitors = f"{custom_competitors} {default_competitors}".strip()

    # Step 1a: Broad discovery
    discovery_results = tavily.search(
        query=f"India fintech credit card {feature_short} 2024",
        max_results=3,
        search_depth="basic"
    )

    # Step 1b: Competitor targeted search
    competitor_results = tavily.search(
        query=f"{all_competitors} {feature_short} India",
        max_results=5,
        search_depth="advanced"
    )

    # Step 1c: Legacy bank search
    legacy_results = tavily.search(
        query=f"ICICI Axis Kotak SBI {feature_short} credit card India feature",
        max_results=3,
        search_depth="basic"
    )

    # Step 1d: Community signal — Reddit and Quora
    community_results = tavily.search(
        query=f"reddit OR quora {feature_short} credit card India review complaint experience 2024",
        max_results=3,
        search_depth="basic"
    )

    # Step 1e: YouTube signal
    youtube_results = tavily.search(
        query=f"youtube {feature_short} credit card India review explained 2024",
        max_results=2,
        search_depth="basic"
    )
    

   # Deduplicate all results
    all_results = (
        discovery_results.get('results', []) +
        competitor_results.get('results', []) +
        legacy_results.get('results', []) +
        community_results.get('results', []) +
        youtube_results.get('results', [])
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

    # Step 2: RBI or global context
    if is_rbi_feature:
        secondary_context = extract_rbi_context(enriched_description)
        secondary_label = "RBI REGULATORY CONTEXT"
    else:
        global_results = tavily.search(
            query=f"global best practice {feature_short} credit card fintech Monzo Revolut 2024",
            max_results=3,
            search_depth="basic"
        )
        secondary_context = "\n\n".join([
            f"Source: {r['url']}\n{r['content']}"
            for r in global_results.get('results', [])
        ])
        secondary_label = "GLOBAL BEST PRACTICES"

    # Step 3: Build prompt
    prompt = f"""A PM has asked for competitive intelligence on this feature:

FEATURE NAME: {input.feature_name}
FEATURE DESCRIPTION: {input.feature_description}
{qa_context}

IMPORTANT RULES:
- HDFC Bank, HDFC PIXEL Play, and PayZapp are the products we are BUILDING — never list them as competitors
- Use ONLY the search results below — do not use training memory
- If something is not in sources, say "not found in current sources"
- Competitor priority: Digital-first cards (OneCard, Scapia, Slice) → Legacy banks (ICICI, Axis, Kotak) → NBFCs/Fintechs (KreditBee, Freo)

SEARCH RESULTS:
{competitor_context}

{secondary_label}:
{secondary_context}

Respond in exactly this structure:

COMPETITORS:
List up to 5. Tag each with priority: [DIGITAL-FIRST] [LEGACY BANK] or [FINTECH/NBFC]
For each:
- Name | Product name [PRIORITY TAG]
- Why relevant: (1 sentence)
- What they built: (2 sentences — specific, not generic)
- One key point: (1 sentence)
- Confidence: High (has URL) / Medium (inferred) / Low (assumed)
- Source: (URL or "not found")

WHITESPACE SIGNAL:
If fewer than 2 competitors found with this specific feature:
"FIRST-MOVER OPPORTUNITY: [explain the gap in 1 sentence]"
Otherwise skip.

WHAT REAL USERS SAY:
2-3 observations from Reddit or app store reviews with source URLs.
If none found: "No community signal found in current sources."

{secondary_label} SUMMARY:
If RBI: Quote exact RBI rule. List what PIXEL must comply with as bullet points.
If global: 2-3 examples, what works, why, source URL.

DIFFERENTIATION HYPOTHESES:
Generate exactly 3 differentiation angles for PIXEL. 

CRITICAL RULE: Each angle must represent a fundamentally different strategic choice — different positioning, different target user segment, different build complexity, or different competitive moat. A PM should feel genuine tension choosing between them. They must NOT be sequential steps of the same feature.

Think across these dimensions for contrast:
- Simple vs sophisticated (zero-UI automation vs intelligent personalisation)
- Defensive vs offensive (compliance-minimum vs market leadership)
- Transactional vs relational (solve the immediate problem vs create a deeper relationship moment)
- Fast to ship vs high moat (quick win vs durable advantage)

For each angle:
- Angle name: (3-4 words)
- What to build: (1 sentence — specific, not generic)
- Strategic logic: (1 sentence — why this is a distinct bet)
- RBI compliance: (1 sentence)
- Build complexity: Low / Medium / High
- Ops impact: (1 sentence)

RECOMMENDED ANGLE:
Pick the strongest angle for PIXEL's current stage and constraints.
- Which angle: (name it)
- Why now: (why this angle fits PIXEL's current technical maturity and team capacity)
- Why not the others: (1 sentence each explaining the tradeoff)

OVERALL CONFIDENCE: High / Medium / Low
Reason: (1 sentence)"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=3000,
        system=[
            {
                "type": "text",
                "text": f"You are a senior competitive intelligence analyst for HDFC PIXEL Studio. HDFC Bank, PIXEL Play, and PayZapp are the products being built — never list them as competitors.\n\nPIXEL Studio context:\n{pixel_context}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": prompt}]
    )

    result = {
        "agent": "competitor",
        "feature_name": input.feature_name,
        "timestamp": datetime.now().isoformat(),
        "sources_searched": len(unique_results),
        "rbi_triggered": is_rbi_feature,
        "qa_history": input.qa_history,
        "response": message.content[0].text
    }

    # Save analysis
    saved_path = save_analysis(input.feature_name, result)
    result["saved_to"] = saved_path

    return result
# ============================================================
# CS AGENT
# ============================================================

class CSInput(BaseModel):
    feature_name: str
    feature_description: str
    competitors: str = ""
    qa_history: list = []


def load_prd_context(feature_name: str) -> str:
    """Find and extract most relevant PRD from /data/prds/ based on feature name."""
    from docx import Document

    prds_dir = Path("data/prds")
    if not prds_dir.exists():
        return "No PRDs found in data/prds directory."

    # Find all docx files
    prd_files = list(prds_dir.glob("*.docx"))
    if not prd_files:
        return "No PRD files found."

    # Score each PRD by keyword match with feature name
    feature_keywords = [w.lower() for w in feature_name.split()
                        if len(w) > 3]

    best_file = None
    best_score = 0

    for prd_file in prd_files:
        filename_lower = prd_file.name.lower()
        score = sum(1 for kw in feature_keywords if kw in filename_lower)
        if score > best_score:
            best_score = score
            best_file = prd_file

    # If no keyword match, use first PRD
    if not best_file:
        best_file = prd_files[0]

    # Extract text from docx
    try:
        doc = Document(best_file)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return f"PRD: {best_file.name}\n\n{text[:4000]}"
    except Exception as e:
        return f"Could not read PRD: {str(e)}"


@app.post("/cs-agent")
def cs_agent(input: CSInput):

    pixel_context = load_pixel_context()
    qa_context = format_qa_history(input.qa_history)
    enriched_description = f"{input.feature_description}\n{qa_context}"
    feature_short = input.feature_name[:100]

    # Load relevant PRD
    prd_context = load_prd_context(input.feature_name)

    # Detect RBI feature
    is_rbi_feature = any(word in enriched_description.lower()
                         for word in ['rbi', 'regulatory', 'compliance',
                                      'guideline', 'circular', 'mandate',
                                      'refund', 'bbps', 'autopay'])

    # Search 1: Competitor CX handling
    cx_results = tavily.search(
        query=f"OneCard Scapia Slice {feature_short} customer support FAQ self-serve India",
        max_results=3,
        search_depth="basic"
    )

    # Search 2: Community signal — what users complain about
    complaint_results = tavily.search(
        query=f"reddit quora {feature_short} credit card India problem complaint support 2024",
        max_results=3,
        search_depth="basic"
    )

    # Search 3: Global CX best practice
    global_cx_results = tavily.search(
        query=f"best practice {feature_short} credit card customer support deflection self-serve",
        max_results=2,
        search_depth="basic"
    )

    # Format search results
    cx_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in cx_results.get('results', [])
    ])

    complaint_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in complaint_results.get('results', [])
    ])

    global_cx_context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in global_cx_results.get('results', [])
    ])

    # RBI context if applicable
    rbi_context = ""
    if is_rbi_feature:
        rbi_context = f"\nRBI REGULATORY CONTEXT:\n{extract_rbi_context(enriched_description)}"

    # Build prompt
    prompt = f"""You are a senior customer support strategist reviewing a PRD for HDFC PIXEL Studio.

FEATURE NAME: {input.feature_name}
FEATURE DESCRIPTION: {input.feature_description}
{qa_context}

PRD CONTENT:
{prd_context}

{rbi_context}

COMPETITOR CX SEARCH RESULTS:
{cx_context}

USER COMPLAINT SIGNAL:
{complaint_context}

GLOBAL CX BEST PRACTICES:
{global_cx_context}

Your job is to identify every CS impact this feature will have and how to handle it.

IMPORTANT RULES:
- Be specific to this feature — no generic CS advice
- Reference the PRD content where relevant
- Flag gaps in the PRD that will cause CS problems
- Use search results to benchmark competitor CX approaches

Respond in exactly this structure:

CONTACT DRIVERS:
List every reason a user will contact CS because of this feature.
For each:
- Scenario: (what happened from user's perspective)
- Likelihood: High / Medium / Low
- PRD gap: (is this covered in the PRD? Yes / Partial / No)

COMPETITOR CX BENCHMARK:
For each high-likelihood contact driver:
- How does the best competitor handle this? (specific, not generic)
- Self-serve or agent-assisted?
- What PIXEL should copy or improve on
- Source URL

FAQ COVERAGE:
For each contact driver, write the FAQ:
- Question: (exactly how a user would phrase it)
- Answer: (what the CS agent should say — specific, actionable)
- Escalation needed: Yes / No

AGENT CENTRE REQUIREMENTS:
What information must be visible to CS agents to resolve queries?
List as bullet points. Flag any that are NOT currently in the PRD.

ERROR SCREEN AUDIT:
List every error state this feature introduces.
For each:
- Error scenario
- Current PRD handling: (documented / not documented)
- Recommended next action for user

ESCALATION PATHS:
Which scenarios cannot be resolved by tier-1 CS?
For each:
- Scenario
- Who handles it (tier-2 / bank team / Zeta ops)
- Expected resolution time

TICKET DEFLECTION OPPORTUNITIES:
What self-serve features would eliminate CS contact entirely?
For each opportunity:
- What to build
- Estimated ticket reduction %
- Complexity: Low / Medium / High

OVERALL CS READINESS SCORE: X/10
Reason: (2 sentences — what's good, what's missing)"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=3000,
        system=[
            {
                "type": "text",
                "text": f"You are a senior customer support strategist for HDFC PIXEL Studio. You review PRDs to identify CS impact, gaps, and deflection opportunities.\n\nPIXEL Studio context:\n{pixel_context}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": prompt}]
    )

    result = {
        "agent": "cs",
        "feature_name": input.feature_name,
        "timestamp": datetime.now().isoformat(),
        "prd_used": prd_context.split('\n')[0],
        "rbi_triggered": is_rbi_feature,
        "qa_history": input.qa_history,
        "response": message.content[0].text
    }

    saved_path = save_analysis(input.feature_name + "_cs", result)
    result["saved_to"] = saved_path

    return result