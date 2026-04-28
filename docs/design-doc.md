# PRD Agent Studio — Design Document
## Competitor Intelligence Agent v1.0

**Author:** Karthik Raman, Associate Director of Products, Zeta India  
**Built:** April–May 2026  
**Stack:** Python 3.11, FastAPI, Anthropic Claude Sonnet 4.5, Tavily Search API,  
           PyMuPDF (RBI PDF extraction), M365 Copilot (planned integration)  
**Repo:** github.com/kirru261990/prd-agent-studio  

---

## 1. Problem I was solving

PIXEL Studio ships 40+ PRDs per year. Every PRD requires competitive 
context — who has built this feature, how they positioned it, what 
customers say, and how PIXEL can differentiate.

Before this agent, this research was done manually by the PM:
- Google searches across 5-6 sources
- Manual RBI circular lookup for regulatory features  
- No structured differentiation framework
- Time taken: 2-3 hours per feature
- Quality: inconsistent, dependent on PM's bandwidth and familiarity 
  with the competitive landscape

The agent reduces this to 2-3 minutes with higher consistency and 
structured output that feeds directly into the PRD.

---

## 2. Why an agent, not a prompt

A single prompt would return generic output because it has no access 
to real-time information, no ability to ask clarifying questions, and 
no way to extract context from RBI regulatory documents.

This is agentic because it:
- Makes autonomous decisions about which search queries to run based 
  on feature type (RBI vs non-RBI)
- Decides when it has enough clarifying context to proceed vs when 
  to ask another question
- Retrieves relevant RBI circular sections dynamically based on 
  feature keywords
- Combines multiple information sources (web search, PDF extraction, 
  community signal) before reasoning
- Detects whitespace opportunities when no competitor data is found
- Saves every analysis automatically without being asked

Each of these is a decision the agent makes — not the user.

---

## 3. Architecture decisions and why

**FastAPI over Flask:**  
FastAPI is async-ready, has automatic API documentation, and uses 
Pydantic for input validation. For a PM learning to build, the 
automatic /docs endpoint at localhost:8000/docs is invaluable for 
testing without a UI.

**Tavily over SerpAPI:**  
Tavily is purpose-built for AI agents — it returns clean structured 
content, not raw HTML. SerpAPI returns links; Tavily returns readable 
text that Claude can reason over directly.

**Prompt caching for PIXEL context:**  
The PIXEL Studio context document (~500 tokens) is injected into 
every API call. Without caching, this costs tokens on every request. 
With Anthropic's ephemeral cache, the context is cached for 5 minutes 
and costs 90% less on repeated calls within a session.

**Separate /clarify and /competitor-agent endpoints:**  
The clarifying questions step uses Claude Sonnet (fast, cheap) because 
it only needs to generate one sharp question. The analysis step uses 
the same model but with a much larger context window and more complex 
reasoning. Separating them keeps the UI responsive and allows 
independent model tuning per step.

**RBI PDF extraction over web search for regulatory context:**  
Web search returns blog posts summarising RBI guidelines — often 
incomplete or outdated. Extracting directly from the official RBI 
Master Direction PDFs gives verbatim regulatory text that can be 
quoted in PRDs without citation risk.

**Three-tier competitor search:**  
Digital-first cards (OneCard, Scapia, Slice) are searched first 
because they are PIXEL's primary competitive set. Legacy banks 
(ICICI, Axis, Kotak) are searched second. NBFCs and fintechs 
(KreditBee, Freo) third. This tiering ensures the most relevant 
competitors surface at the top of the output.

---

## 4. What I built — week by week

**Week 0:** Mac environment setup, Python 3.11, virtual environment, 
Anthropic API connected, GitHub repo live, RBI PDFs collected locally.

**Week 1 (Days 1-3):** FastAPI server skeleton, first Claude API call 
working end to end, browser UI built in HTML, CORS issue debugged and 
fixed, basic competitor agent returning structured output.

**Week 1 (Days 4-5):** Tavily web search integrated, two-query search 
strategy (competitor + community signal), PIXEL context document 
written and injected via prompt caching, output structure tightened 
with explicit format constraints.

**Week 2:** RBI PDF extraction added using PyMuPDF, three-tier 
competitor search architecture, whitespace detection, HDFC/PIXEL 
exclusion rule, dynamic search query based on feature type, output 
persistence to JSON, two-step UI with clarifying questions flow, 
multi-hypothesis differentiation with recommended angle.

---

## 5. What failed and what I learned

**Failure 1 — Generic competitor output in early versions:**  
The first version hardcoded "ICICI Axis SBI" into the search query. 
This returned legacy bank marketing pages, not competitive intelligence. 
Learning: competitor discovery must be dynamic, starting broad and 
filtering by relevance, not hardcoded.

**Failure 2 — Global best practices section was useless:**  
Searching "global best practice + feature name" returned random fintech 
blog posts with no relevance to Indian credit card context. 
Learning: global search only adds value for non-regulatory, 
non-India-specific features. For RBI-mandated features, the 
regulatory text itself is more valuable than global comparisons.

**Failure 3 — Differentiation angles were sequential not distinct:**  
Early versions produced 3 angles that were actually phases of the same 
feature — notification, then ML, then instant transfer. A PM couldn't 
choose between them because they weren't competing strategic bets. 
Learning: prompt must explicitly instruct Claude to generate angles 
across different strategic dimensions — defensive vs offensive, 
fast-to-ship vs high-moat, transactional vs relational.

**Failure 4 — HDFC listed as a competitor:**  
The search returned HDFC's own product pages and Claude listed PIXEL 
Play as a competitor to itself. Learning: system prompt must explicitly 
exclude the product being built from the competitor list.

**Failure 5 — Indentation errors from manual nano editing:**  
Multiple sessions of editing server.py in nano resulted in broken 
Python indentation. Learning: always use VS Code for code editing, 
never nano for anything beyond 5 lines.

---

## 6. Eval results

**Feature 1: Instant card freeze and unfreeze**  
What worked: Correct digital-first competitors identified, 
differentiation angle was PIXEL-specific  
What didn't: Available since data was rarely found, customer 
sentiment was generic

**Feature 2: Credit balance refund (RBI mandated)**  
What worked: RBI verbatim text extracted correctly, whitespace 
signal fired correctly, Reddit user frustration captured accurately, 
three distinct differentiation angles  
What didn't: Legacy bank (ICICI) appeared before digital-first 
competitors because ICICI had more indexed content on this feature

**Feature 3: Smart EMI — convert purchases without merchant route**  
What worked: Competitor discovery improved with digital-first focus, 
PIXEL rewards structure referenced in differentiation  
What didn't: Global best practices section returned irrelevant results 
for India-specific EMI products

---

## 7. What I would build next

**Improvement 1 — Persistent competitor database:**  
Every analysis currently starts from zero. A JSON-based competitor 
profile store would accumulate intelligence over time — OneCard's 
features, Scapia's positioning, Slice's pricing — updated with each 
query and reused as grounding context in future analyses.

**Improvement 2 — Confidence validation enforcement:**  
Currently confidence level is Claude's opinion. It should be 
rule-based: High only if source URL is provided and URL returns 200. 
Medium if inferred from partial sources. Low if assumed. This 
eliminates overconfident hallucinations.

**Improvement 3 — App store review integration:**  
Google Play reviews for OneCard, Scapia, Slice contain the most 
honest customer signal. Tavily doesn't index these well. A dedicated 
Google Play scraper or the SerpAPI app store endpoint would 
dramatically improve community signal quality.

---

## 8. Director-level reflection

**One sentence business case:**  
"PIXEL Studio ships 40+ PRDs per year — this agent reduces competitive 
research from 2-3 hours to 2-3 minutes per feature, with higher 
consistency and built-in RBI compliance checking, compounding across 
the entire team."

**Data approvals:**  
No approvals needed for the prototype phase — 5-10 real PRDs 
downloaded locally are sufficient for development and demo. 
Post-demo, the path to production requires SharePoint and Confluence 
integration approvals so the agent can read PRDs directly rather than 
requiring manual uploads.

**IT infrastructure to move from localhost to internal access:**  
Three things are needed to move from localhost to team-accessible:
1. An Azure VM or Azure App Service to host the FastAPI server 
   (Zeta likely already has Azure tenant access via M365)
2. A registered Azure AD app so the server can authenticate against 
   SharePoint and read PRD documents via Microsoft Graph API
3. IT approval to whitelist the Anthropic API endpoint as an 
   outbound connection from Zeta's network, or alternatively 
   swap to Azure OpenAI which is already inside the Microsoft 
   trust boundary and requires no new firewall rules

The simplest path: host on Azure App Service, use Azure OpenAI 
instead of Anthropic API, connect to SharePoint via Graph API. 
All three are within Zeta's existing Microsoft enterprise agreement.

**What would convince Sharath to greenlight this:**  
A live demo that produces a genuinely high-quality competitive 
analysis (>80% accuracy vs manual research) in under 30 minutes 
for a real PIXEL Studio feature from the current roadmap. 
The business case is simple: if 5 PMs each spend 2-3 hours per 
PRD on competitive research, and PIXEL ships 40+ PRDs per year, 
that is 400-600 PM hours per year. This agent compresses that 
to under 5 hours total. The ask is not budget — it is IT approval 
for SharePoint integration and Azure hosting.

---

## 9. Agents still to build

**CS Agent:**  
Reads a PRD and identifies customer support impact — new FAQs needed, 
agent training requirements, escalation scenarios, and ticket 
deflection opportunities. Grounded in PIXEL's existing CS process 
documents and historical ticket categories.

**Compliance Agent:**  
Given a PRD feature, checks it against RBI guidelines, flags 
compliance gaps, cites specific circular sections, and outputs a 
structured compliance checklist with confidence levels. Uses the same 
RBI PDF extraction architecture built for the competitor agent, 
extended with a larger circular corpus and stricter citation 
enforcement.

---

## 10. Architecture diagram

User (Teams / Browser UI)
↓
FastAPI Server (localhost → Azure hosting)
↓
┌──────────────────────────────────┐
│  /clarify endpoint               │
│  Claude Sonnet — generates       │
│  sharp clarifying questions      │
│  based on feature description    │
└──────────────────────────────────┘
↓ (after max 3 questions)
┌──────────────────────────────────┐
│  /competitor-agent endpoint      │
│                                  │
│  Tavily Search (4 queries):      │
│  - Broad discovery               │
│  - Digital-first competitors     │
│  - Legacy banks                  │
│  - Community signal              │
│                                  │
│  RBI PDF extraction (PyMuPDF):   │
│  - Only for regulatory features  │
│  - Verbatim circular text        │
│                                  │
│  Claude Sonnet:                  │
│  - PIXEL context (cached)        │
│  - Structured analysis prompt    │
│  - 3 distinct angles + rec       │
│                                  │
│  Output persistence:             │
│  - Saved as JSON locally         │
└──────────────────────────────────┘
↓
Structured intelligence report
(Competitors / Whitespace /
User signal / RBI /
Differentiation / Confidence)

---

*Last updated: April 2026*  
*Next: CS Agent — customer support impact analysis*