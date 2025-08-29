import json
import os
import re
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

# Optional Exa client
try:
	from exa_py import Exa  # type: ignore
except Exception:
	Exa = None  # type: ignore

from schemas import (
	CompaniesPayload,
	Company,
	ContactsPayload,
	ContactsPerCompany,
	Contact,
	ResearchPayload,
	ResearchPerCompany,
	EmailsPayload,
)

# -----------------------------
# Logging
# -----------------------------

_logger_initialized = False

def get_logger() -> logging.Logger:
	global _logger_initialized
	logger = logging.getLogger("livetrend_outreach")
	if not _logger_initialized:
		level = os.getenv("LIVETREND_LOG_LEVEL", "INFO").upper()
		logging.basicConfig(level=getattr(logging, level, logging.INFO), format='[%(levelname)s] %(message)s')
		_logger_initialized = True
	return logger


def log_info(message: str) -> None:
	get_logger().info(message)


# -----------------------------
# Generic helpers
# -----------------------------

def response_text(res: Any) -> str:
	"""Best-effort to get text out of an Agno Agent result."""
	for attr in ("text", "output", "response", "content"):
		val = getattr(res, attr, None)
		if isinstance(val, str) and val.strip():
			return val
	if isinstance(res, dict):
		for key in ("content", "text", "output"):
			val = res.get(key)
			if isinstance(val, str) and val.strip():
				return val
	try:
		return str(res)
	except Exception:
		return ""


def parse_json_safe(text: str, default: Any) -> Any:
	if not text:
		return default
	try:
		return json.loads(text)
	except Exception:
		pass
	match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
	if match:
		candidate = match.group(1)
		try:
			return json.loads(candidate)
		except Exception:
			return default
	return default


def ensure_env(openai_key: str, exa_key: str) -> None:
	if openai_key:
		os.environ["OPENAI_API_KEY"] = openai_key
	if exa_key:
		os.environ["EXA_API_KEY"] = exa_key


def payload_to_df(payload: BaseModel) -> pd.DataFrame:
	data = payload.model_dump()
	if "companies" in data and isinstance(data["companies"], list):
		if data["companies"] and isinstance(data["companies"][0], dict) and "contacts" in data["companies"][0]:
			rows = []
			for pc in data["companies"]:
				for ct in pc.get("contacts", []):
					row = {**ct}
					row["company"] = pc.get("company")
					rows.append(row)
			return pd.DataFrame(rows)
		elif data["companies"] and isinstance(data["companies"][0], dict) and "insights" in data["companies"][0]:
			rows = []
			for rc in data["companies"]:
				for ins in rc.get("insights", []):
					rows.append({"company": rc.get("company"), "insight": ins})
			return pd.DataFrame(rows)
		else:
			return pd.DataFrame(data["companies"])
	if "emails" in data:
		return pd.DataFrame(data["emails"])
	return pd.DataFrame()


def save_exports(out_dir: str, prefix: str, payload_dict: Dict[str, Any]) -> Dict[str, str]:
	os.makedirs(out_dir, exist_ok=True)
	ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
	paths: Dict[str, str] = {}
	json_path = os.path.join(out_dir, f"{prefix}-{ts}.json")
	with open(json_path, "w", encoding="utf-8") as f:
		f.write(json.dumps(payload_dict, ensure_ascii=False, indent=2))
	paths["json"] = json_path
	try:
		import pandas as pd
		df = pd.DataFrame()
		if "companies" in payload_dict and isinstance(payload_dict["companies"], list):
			if payload_dict["companies"] and isinstance(payload_dict["companies"][0], dict) and "contacts" in payload_dict["companies"][0]:
				rows = []
				for pc in payload_dict["companies"]:
					for ct in pc.get("contacts", []):
						row = {**ct}
						row["company"] = pc.get("company")
						rows.append(row)
				df = pd.DataFrame(rows)
			elif payload_dict["companies"] and isinstance(payload_dict["companies"][0], dict) and "insights" in payload_dict["companies"][0]:
				rows = []
				for rc in payload_dict["companies"]:
					for ins in rc.get("insights", []):
						rows.append({"company": rc.get("company"), "insight": ins})
				df = pd.DataFrame(rows)
			else:
				df = pd.DataFrame(payload_dict["companies"]) if payload_dict["companies"] else pd.DataFrame()
		elif "emails" in payload_dict:
			df = pd.DataFrame(payload_dict["emails"]) if payload_dict["emails"] else pd.DataFrame()
		csv_path = os.path.join(out_dir, f"{prefix}-{ts}.csv")
		df.to_csv(csv_path, index=False)
		paths["csv"] = csv_path
	except Exception:
		pass
	return paths


def with_retries(fn: Callable[[], Any], retries: int = 2) -> Any:
	last_exc = None
	for _ in range(max(1, retries + 1)):
		try:
			return fn()
		except Exception as e:
			last_exc = e
	return fn() if last_exc is None else (_ for _ in ()).throw(last_exc)


# -----------------------------
# Exa fallback helpers
# -----------------------------

def _get_exa() -> Optional[Any]:
	if Exa is None:
		return None
	key = os.getenv("EXA_API_KEY", "")
	if not key:
		return None
	try:
		return Exa(key)
	except Exception:
		return None


def _domain(url: str) -> str:
	m = re.search(r"https?://([^/]+)", url)
	return m.group(1) if m else url


def _brand_from_domain(domain: str) -> str:
	parts = domain.split(".")
	parts = [p for p in parts if p and p.lower() != "www"]
	if len(parts) >= 2:
		return parts[-2].capitalize()
	return parts[0].capitalize() if parts else domain


def _infer_company_name(title: str, url: str) -> str:
	if title:
		clean = re.sub(r"\s*[-|•].*$", "", title).strip()
		if clean and clean.lower() not in {"home", "welcome"} and not re.search(r"cookie|privacy|terms|login", clean, re.I):
			return clean
	d = _domain(url)
	return _brand_from_domain(d)

_BAD_SNIPPET_RE = re.compile(r"cookie|privacy|terms|javascript|enable|old browser|support|consent", re.I)


def _clean_why_fit(text: str, fallback: str) -> str:
	if not text or _BAD_SNIPPET_RE.search(text):
		return fallback
	return text.strip()[:160]

_FASHION_WORDS = re.compile(r"trend|collection|assort|buying|merchand|retail|fashion|brand|e-?commerce|pricing|sell|SKU|inventory|season", re.I)


def _extract_key_points(text: str, brand: Optional[str] = None) -> List[str]:
	sentences = re.split(r"(?<=[.!?])\s+", text or "")
	points: List[str] = []
	for s in sentences:
		ls = s.lower()
		if brand and brand.lower() not in ls and not _FASHION_WORDS.search(ls):
			continue
		if 40 <= len(s) <= 200:
			points.append(s.strip())
	return points[:4]


def _excluded(domain: str, exclude_domains: List[str]) -> bool:
	domain = domain.lower()
	if any(domain.endswith(ex.lower()) for ex in exclude_domains):
		return True
	return "livetrend" in domain


def discover_companies_via_exa(target_desc: str, offering_desc: str, limit: int, exclude_domains: Optional[List[str]] = None, debug: bool = True) -> CompaniesPayload:
	exclude_domains = exclude_domains or []
	exa = _get_exa()
	if exa is None:
		return CompaniesPayload()
	query = (
		f"(fashion retailer OR apparel brand) site:*.com (buying OR merchandising OR ecommerce) -site:livetrend.co -cookie -privacy -terms -login -support — {target_desc}"
	)
	if debug:
		log_info(f"[EXA] search companies query: {query[:140]}…")
	try:
		results = exa.search_and_contents(query=query, num_results=limit * 2, use_autoprompt=True)
		items = results.results if hasattr(results, "results") else results
	except Exception:
		try:
			items = exa.search(query=query, num_results=limit * 2, use_autoprompt=True)
			items = items.results if hasattr(items, "results") else items
		except Exception:
			return CompaniesPayload()
	companies: List[Company] = []
	seen: set = set()
	for r in items:
		url = getattr(r, "url", None) or r.get("url", "")
		title = getattr(r, "title", None) or r.get("title", "")
		text = getattr(r, "text", None) or r.get("text", "")
		domain = _domain(url)
		if _excluded(domain, exclude_domains) or domain in seen:
			continue
		seen.add(domain)
		name = _infer_company_name(title, url)
		why = _clean_why_fit(text or title, fallback="Retailer/brand with ecommerce; matches targeting.")
		if not _FASHION_WORDS.search((text or title or "")):
			continue
		companies.append(Company(name=name, website=url, why_fit=why))
		if len(companies) >= limit:
			break
	return CompaniesPayload(companies=companies)


def find_contacts_via_exa(companies: CompaniesPayload, debug: bool = True) -> ContactsPayload:
	exa = _get_exa()
	if exa is None:
		return ContactsPayload()
	roles = [
		"Head of Buying",
		"Merchandising Director",
		"VP Merchandising",
		"GTM Lead",
		"Partnerships Manager",
		"Product Marketing",
		"Founder",
	]
	out: List[ContactsPerCompany] = []
	for c in companies.companies:
		domain = _domain(c.website)
		company_brand = _brand_from_domain(domain)
		contacts: List[Contact] = []
		for role in roles:
			q = f"{role} {company_brand} email OR contact"
			if debug:
				log_info(f"[EXA] search contact query: {q}")
			try:
				res = exa.search(query=q, num_results=2, use_autoprompt=True)
				items = res.results if hasattr(res, "results") else res
				for r in items:
					title = getattr(r, "title", None) or r.get("title", "")
					url = getattr(r, "url", None) or r.get("url", "")
					person = _infer_person_from_title(title)
					if person:
						contacts.append(Contact(name=person, role=role, email=_infer_email(person, domain), inferred=True, source=url))
			except Exception:
				continue
			if len(contacts) >= 2:
				break
		out.append(ContactsPerCompany(company=c.name, contacts=contacts[:2]))
	return ContactsPayload(companies=out)


def collect_research_via_exa(companies: CompaniesPayload, debug: bool = True) -> ResearchPayload:
	exa = _get_exa()
	if exa is None:
		return ResearchPayload()
	out: List[ResearchPerCompany] = []
	for c in companies.companies:
		domain = _domain(c.website)
		brand = _brand_from_domain(domain)
		insights: List[str] = []
		queries = [f"site:{domain} (about OR team OR blog)", f"site:reddit.com {brand}"]
		for q in queries:
			if debug:
				log_info(f"[EXA] search research query: {q}")
			try:
				res = exa.search_and_contents(query=q, num_results=3, use_autoprompt=True)
				items = res.results if hasattr(res, "results") else res
				for r in items:
					text = getattr(r, "text", None) or r.get("text", "")
					for pt in _extract_key_points(text, brand=brand):
						if pt not in insights:
							insights.append(pt)
			except Exception:
				continue
		out.append(ResearchPerCompany(company=c.name, insights=insights[:4]))
	return ResearchPayload(companies=out)


# -----------------------------
# Normalization helpers
# -----------------------------

def normalize_companies(payload: CompaniesPayload, exclude_domains: Optional[List[str]] = None) -> CompaniesPayload:
	exclude_domains = exclude_domains or []
	clean: List[Company] = []
	seen: set = set()
	for c in payload.companies:
		d = _domain(c.website)
		if _excluded(d, exclude_domains) or d in seen:
			continue
		seen.add(d)
		name = _infer_company_name(c.name, c.website)
		why = _clean_why_fit(c.why_fit, fallback="Retailer/brand with ecommerce; matches targeting.")
		if not _FASHION_WORDS.search(c.why_fit + " " + name):
			continue
		clean.append(Company(name=name, website=c.website, why_fit=why))
	return CompaniesPayload(companies=clean)


def clean_research(payload: ResearchPayload) -> ResearchPayload:
	out: List[ResearchPerCompany] = []
	for rc in payload.companies:
		filtered = [p for p in rc.insights if not _BAD_SNIPPET_RE.search(p) and _FASHION_WORDS.search(p)]
		unique: List[str] = []
		seen: set = set()
		for p in filtered:
			k = p.strip().lower()
			if k not in seen:
				seen.add(k)
				unique.append(p)
		out.append(ResearchPerCompany(company=rc.company, insights=unique[:4]))
	return ResearchPayload(companies=out)


# -----------------------------
# Email fallback generator
# -----------------------------

def generate_template_emails(companies: CompaniesPayload, contacts: ContactsPayload, research: ResearchPayload, sender: Dict[str, str], style: str = "Professional") -> EmailsPayload:
	emails: List[Dict[str, str]] = []
	style_hint = {
		"Professional": "ton professionnel, clair et concis",
		"Casual": "ton amical et conversationnel",
		"Cold": "ton direct et orienté ROI",
		"Consultative": "ton consultatif et orienté diagnostic",
	}.get(style, "ton professionnel")
	company_to_contacts = {c.company: c.contacts for c in contacts.companies}
	research_map = {r.company: r.insights for r in research.companies}
	for comp in companies.companies:
		targets = company_to_contacts.get(comp.name) or [Contact(name="Équipe", role="Team")]
		brand = _brand_from_domain(_domain(comp.website))
		for ct in targets:
			subject = f"{brand} — booster vos décisions d'achat avec la data"
			ins = [p for p in (research_map.get(comp.name) or []) if not _BAD_SNIPPET_RE.search(p)]
			ins_text = ("\n- " + "\n- ".join(ins[:2])) if ins else "\n- Observation marché pertinente"
			body = (
				f"Bonjour {ct.name},\n\n"
				f"Je me permets de vous contacter (\u2013 {style_hint}) au nom de {sender.get('company','Livetrend')}. "
				"Nous aidons les équipes buying/merchandising à construire des collections best-sellers grâce à :\n"
				"- analyses de tendances data-driven\n- veille concurrentielle en temps réel\n- planification d'assortiment et pricing\n\n"
				f"Ce qui a retenu notre attention pour {brand} :{ins_text}\n\n"
				"Ouvert à 15 min pour voir comment ces données peuvent sécuriser vos décisions ?\n\n"
				f"Bien à vous,\n{sender.get('name','')} — {sender.get('company','Livetrend')}"
			)
			emails.append({
				"company": comp.name,
				"contact": ct.name,
				"subject": subject,
				"body": body,
			})
	return EmailsPayload(emails=[e for e in emails])
