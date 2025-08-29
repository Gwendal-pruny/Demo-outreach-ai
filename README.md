## Livetrend Outreach Demo (Streamlit)

A user-friendly, ready-to-run demo that builds personalized B2B outreach emails using a multi-agent pipeline per the Unwind AI tutorial.

- Reference: `Build an AI Email GTM Outreach Agent Team` ([theunwindai.com](https://www.theunwindai.com/p/build-an-ai-email-gtm-outreach-agent-team))
- Entry points:
  - `ai_email_gtm_outreach_agent.py` (Agno-based, faithful to the tutorial)
  - `livetrend_outreach_demo.py` (simple baseline without Agno)

### Features
- Company discovery with Exa AI (via Agno ExaTools)
- Contact role inference with email inference flag
- Research from company sites and Reddit
- Personalized email generation (Professional, Casual, Cold, Consultative)
- Streamlit UI with progress and results sections
- Display-only output (no sending of emails)
- Model selector for each agent stage (gpt-5, gpt-4o, gpt-4o-mini)
- One-click JSON downloads for companies/contacts/research/emails
- Pydantic schema validation for all stages
- Advanced filters: industries, regions, exclude domains
- Optional follow-up sequence generation per contact
- Optional quality-check agent (QC) for email review
- Auto-save JSON/CSV to `out/` with timestamps

### Prerequisites
- Python 3.10+
- API keys:
  - OpenAI (`OPENAI_API_KEY`)
  - Exa AI (`EXA_API_KEY`)

### Quickstart (Windows PowerShell)
```powershell
# From project root
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the Agno-based app (faithful to tutorial)
streamlit run ai_email_gtm_outreach_agent.py

# Alternatively, run the baseline app
# streamlit run livetrend_outreach_demo.py
```

### Configure
- Sidebar → API keys and models
- Sidebar → Filters (industries, regions, exclude domains)
- Sidebar → Follow-ups (0–3), QC toggle

### Exports
- UI: Download buttons (JSON)
- Filesystem: JSON & CSV saved to `out/` with timestamps for each stage

### Notes
- This demo performs live web search via Exa AI and content generation via OpenAI. Costs may apply.
- The app does not send emails. All outputs are rendered in the UI only.

### Credits
- Unwind AI tutorial: [theunwindai.com/p/build-an-ai-email-gtm-outreach-agent-team](https://www.theunwindai.com/p/build-an-ai-email-gtm-outreach-agent-team)
