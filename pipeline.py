from typing import Any, Dict, List
from pydantic import ValidationError
import json

from schemas import CompaniesPayload, ContactsPayload, ResearchPayload, EmailsPayload
from utils import parse_json_safe, response_text
from utils import (
	discover_companies_via_exa,
	find_contacts_via_exa,
	collect_research_via_exa,
	log_info,
	generate_template_emails,
	normalize_companies,
	clean_research,
)


def run_company_finder(agent: Any, target_desc: str, offering_desc: str, num_companies: int, retries: int = 1, allow_fallbacks: bool = True) -> CompaniesPayload:
	prompt = (
		"Trouve des entreprises correspondant au ciblage. "
		"Réponds UNIQUEMENT en JSON: {companies: [{name, website, why_fit}]}. Respecte 'limit'.\n\n"
		f"target_companies: {target_desc}\n"
		f"offering: {offering_desc}\n"
		f"limit: {num_companies}\n"
	)
	attempt = 0
	payload = CompaniesPayload()
	while attempt <= max(1, retries):
		log_info(f"[OPENAI] CompanyFinder: prompt sent (attempt {attempt+1})")
		res = agent.run(prompt)
		text = response_text(res)
		data = parse_json_safe(text, {"companies": []})
		try:
			payload = CompaniesPayload(**data)
		except ValidationError:
			payload = CompaniesPayload()
		if payload.companies:
			break
		attempt += 1
	if not payload.companies and allow_fallbacks:
		log_info("[EXA] Fallback: discover companies via Exa")
		payload = discover_companies_via_exa(target_desc, offering_desc, num_companies, exclude_domains=["livetrend.co"])
	# Normalize and filter for quality
	payload = normalize_companies(payload, exclude_domains=["livetrend.co"]) if payload.companies else payload
	return payload


def run_contact_finder(agent: Any, companies: CompaniesPayload, target_desc: str, offering_desc: str, retries: int = 1, allow_fallbacks: bool = True) -> ContactsPayload:
	companies_json = json.dumps([c.model_dump() for c in companies.companies], ensure_ascii=False)
	prompt = (
		"Pour chaque entreprise, trouve 1-2 décideurs pertinents (Founder's Office, GTM, Sales, Partnerships, Product Marketing). "
		"Si email non trouvé, infère et mets inferred=true. Réponds en JSON: {companies: [{company, contacts: [{name, role, email, inferred, source}]}]}.\n\n"
		f"companies: {companies_json}\n"
		f"target_companies: {target_desc}\n"
		f"offering: {offering_desc}\n"
	)
	attempt = 0
	payload = ContactsPayload()
	while attempt <= max(1, retries):
		log_info(f"[OPENAI] ContactFinder: prompt sent (attempt {attempt+1})")
		res = agent.run(prompt)
		text = response_text(res)
		data = parse_json_safe(text, {"companies": []})
		try:
			payload = ContactsPayload(**data)
		except ValidationError:
			payload = ContactsPayload()
		if payload.companies:
			break
		attempt += 1
	if not payload.companies and allow_fallbacks:
		log_info("[EXA] Fallback: find contacts via Exa")
		payload = find_contacts_via_exa(companies)
	return payload


def run_research(agent: Any, companies: CompaniesPayload, retries: int = 1, allow_fallbacks: bool = True) -> ResearchPayload:
	companies_json = json.dumps([c.model_dump() for c in companies.companies], ensure_ascii=False)
	prompt = (
		"Collecte 2-4 insights par entreprise depuis leur site (about/blog/produit) et Reddit. "
		"Réponds en JSON: {companies: [{company, insights: [..]}]}.\n\n"
		f"companies: {companies_json}\n"
	)
	attempt = 0
	payload = ResearchPayload()
	while attempt <= max(1, retries):
		log_info(f"[OPENAI] Research: prompt sent (attempt {attempt+1})")
		res = agent.run(prompt)
		text = response_text(res)
		data = parse_json_safe(text, {"companies": []})
		try:
			payload = ResearchPayload(**data)
		except ValidationError:
			payload = ResearchPayload()
		if payload.companies:
			break
		attempt += 1
	if not payload.companies and allow_fallbacks:
		log_info("[EXA] Fallback: collect research via Exa")
		payload = collect_research_via_exa(companies)
	# Clean insights for fashion relevance
	payload = clean_research(payload) if payload.companies else payload
	return payload


def run_email_writer(agent: Any, companies: CompaniesPayload, contacts: ContactsPayload, research: ResearchPayload, sender: Dict[str, str], style: str = "Professional", retries: int = 1, allow_fallbacks: bool = True) -> EmailsPayload:
	companies_json = json.dumps([c.model_dump() for c in companies.companies], ensure_ascii=False)
	contacts_json = json.dumps(contacts.model_dump(), ensure_ascii=False)
	research_json = json.dumps(research.model_dump(), ensure_ascii=False)
	sender_json = json.dumps(sender, ensure_ascii=False)
	prompt = (
		"Rédige des emails B2B personnalisés (120-160 mots), en intégrant 1-2 lignes de personnalisation depuis les insights. "
		"Utilise sender.name et sender.company pour la signature. Réponds en JSON: {emails: [{company, contact, subject, body, optional followups}]}.\n\n"
		f"companies: {companies_json}\n"
		f"contacts: {contacts_json}\n"
		f"research: {research_json}\n"
		f"sender: {sender_json}\n"
	)
	attempt = 0
	payload = EmailsPayload()
	while attempt <= max(1, retries):
		log_info(f"[OPENAI] EmailWriter: prompt sent (attempt {attempt+1})")
		res = agent.run(prompt)
		text = response_text(res)
		data = parse_json_safe(text, {"emails": []})
		if isinstance(data, dict) and "emails" in data:
			for e in data["emails"]:
				if isinstance(e, dict) and e.get("followups"):
					fus = e.get("followups")
					if isinstance(fus, list) and fus:
						e["body"] = e.get("body", "") + "\n\n---\nFollow-ups:\n" + "\n".join(f"- {t}" for t in fus)
		try:
			for e in data.get("emails", []):
				e.setdefault("company", e.get("company", ""))
				e.setdefault("contact", e.get("contact", ""))
			payload = EmailsPayload(**data)
		except ValidationError:
			payload = EmailsPayload()
		if payload.emails:
			break
		attempt += 1
	if not payload.emails and allow_fallbacks:
		log_info("[TEMPLATE] Fallback: generate simple template emails")
		payload = generate_template_emails(companies, contacts, research, sender, style=style)
	return payload


def run_quality_check(agent: Any, emails: EmailsPayload) -> List[Dict[str, Any]]:
	if not emails.emails:
		return []
	emails_json = json.dumps([e.model_dump() for e in emails.emails], ensure_ascii=False)
	prompt = (
		"Vérifie que chaque email est spécifique et non générique, cite un insight, et évite le spam. "
		"Réponds en JSON: {qc: [{ok: bool, notes: string}]}, ordre identique aux emails.\n\n"
		f"emails: {emails_json}\n"
	)
	res = agent.run(prompt)
	data = parse_json_safe(str(res), {"qc": []})
	items = data.get("qc", [])
	return items if isinstance(items, list) else []
