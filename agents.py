from typing import Any, List

try:
	from agno.agent import Agent
	from agno.memory.v2 import Memory
	from agno.models.openai import OpenAIChat
	from agno.tools.exa import ExaTools
except Exception:
	Agent = None  # type: ignore
	Memory = None  # type: ignore
	OpenAIChat = None  # type: ignore
	ExaTools = None  # type: ignore


def create_company_finder_agent(model_id: str, industries: List[str], regions: List[str], exclude_domains: List[str]) -> Any:
	exa_tools = ExaTools(category="company")
	memory = Memory()
	filters_instruction = (
		f"Industries: {', '.join(industries) if industries else 'any'}. "
		f"Regions: {', '.join(regions) if regions else 'any'}. "
		f"Exclude domains: {', '.join(exclude_domains) if exclude_domains else 'none'}."
	)
	return Agent(
		model=OpenAIChat(id=model_id),
		tools=[exa_tools],
		memory=memory,
		add_history_to_messages=True,
		instructions=[
			"You are CompanyFinderAgent. Use ExaTools to search the web for companies that match the targeting criteria.",
			filters_instruction,
			"Return ONLY valid JSON with key 'companies' as a list; respect the requested limit.",
			"Each item must have: name, website, why_fit (1-2 lines).",
			"Do not include companies whose domains are in the exclude list.",
		],
	)


def create_contact_finder_agent(model_id: str) -> Any:
	exa_tools = ExaTools()
	memory = Memory()
	return Agent(
		model=OpenAIChat(id=model_id),
		tools=[exa_tools],
		memory=memory,
		instructions=[
			"You are ContactFinderAgent. Use ExaTools to find 1-2 relevant decision makers per company.",
			"Prioritize roles from Founder's Office, GTM, Sales leadership, Partnerships, Product Marketing.",
			"If direct emails not found, infer using common formats but mark inferred=true.",
			"Return ONLY valid JSON with companies and contacts.",
		],
	)


def create_research_agent(model_id: str) -> Any:
	exa_tools = ExaTools()
	memory = Memory()
	return Agent(
		model=OpenAIChat(id=model_id),
		tools=[exa_tools],
		memory=memory,
		instructions=[
			"You are ResearchAgent. For each company, collect valuable insights from:",
			"1) Their official website (about, blog, product pages)",
			"2) Reddit discussions (site:reddit.com mentions)",
			"Return 2-4 interesting, non-generic points per company for email personalization.",
		],
	)


def get_email_style_instruction(style_key: str) -> str:
	styles = {
		"Professional": "Write concise, professional B2B outreach emails with a clear value proposition and soft CTA.",
		"Casual": "Write a friendly, conversational email with a single value point and simple question CTA.",
		"Cold": "Write a direct cold email with a strong lead sentence and single-sentence CTA.",
		"Consultative": "Write a consultative email framing 2-3 challenges and a diagnostic-call CTA.",
	}
	return styles.get(style_key, styles["Professional"])


def create_email_writer_agent(model_id: str, style_key: str = "Professional", followups: int = 0) -> Any:
	memory = Memory()
	style_instruction = get_email_style_instruction(style_key)
	followup_instruction = (
		f"Also generate {followups} numbered follow-up emails (1-2 lines each) with JSON key 'followups' as a list."
		if followups > 0 else ""
	)
	return Agent(
		model=OpenAIChat(id=model_id),
		tools=[],
		memory=memory,
		instructions=[
			"You are EmailWriterAgent. Write concise, personalized B2B outreach emails.",
			style_instruction,
			"Length: 120-160 words. Include 1-2 lines of personalization using research insights.",
			followup_instruction,
			"Always use sender.name and sender.company from the input for the signature.",
			"Adapt tone to Livetrend's offering: data-driven trend analysis, competitive intelligence, assortment planning, pricing/buying optimization.",
			"Return ONLY valid JSON with emails: list of {company, contact, subject, body, optional followups}.",
		],
	)


def create_quality_check_agent(model_id: str) -> Any:
	memory = Memory()
	return Agent(
		model=OpenAIChat(id=model_id),
		tools=[],
		memory=memory,
		instructions=[
			"You are QualityCheckAgent. Verify that each email is tailored, specific (non-generic),",
			" references at least one research point, and avoids spammy phrases.",
			"Return ONLY valid JSON with key 'qc' as a list matching emails order, each item with {ok: bool, notes: string}.",
		],
	)
