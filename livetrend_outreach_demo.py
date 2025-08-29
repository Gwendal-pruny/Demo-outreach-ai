import os
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# External clients
try:
    from openai import OpenAI  # openai>=1.0 SDK
except ImportError:
    OpenAI = None  # type: ignore

try:
    from exa_py import Exa  # exa-py package
except ImportError:
    Exa = None  # type: ignore

INSTALL_HELP = "Installez les dépendances avec `pip install -r requirements.txt`."


EMAIL_STYLES: Dict[str, str] = {
    "Professional": (
        "Write concise, professional B2B outreach emails with a clear value proposition, a"
        " tailored insight-based hook, and a soft CTA. Avoid buzzwords."
    ),
    "Casual": (
        "Write a friendly, conversational email with a short intro, one clear value point,"
        " one insight, and a simple question CTA."
    ),
    "Cold": (
        "Write a direct cold email with a sharp subject line, strong lead sentence,"
        " clear ROI angle, and single-sentence CTA."
    ),
    "Consultative": (
        "Write a consultative email framing 2-3 challenges and how we can address them,"
        " referencing insights and ending with a diagnostic call invite."
    ),
}

DEFAULT_TARGET_DESC = (
    "Fashion retailers and apparel brands (EU/US) with e-commerce presence, 50–1000 employees. "
    "Prioritize buying/merchandising-led organizations and fast-fashion or data-driven teams."
)

DEFAULT_OFFERING_DESC = (
    "Livetrend provides data-driven trend analysis and competitive intelligence to design better"
    " collections, plan assortments, and optimize pricing and buying decisions. We analyze 1,040"
    " fashion attributes from e-commerce, shows, and Instagram, map colors to Pantone, and offer"
    " real-time competitive dashboards, trend feeds, and retail search."
)


class Clients:
    def __init__(self, openai_key: str, exa_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError(f"OpenAI n’est pas installé. {INSTALL_HELP}")
        if Exa is None:
            raise RuntimeError(f"Exa-py n’est pas installé. {INSTALL_HELP}")
        self.openai = OpenAI(api_key=openai_key)
        self.exa = Exa(exa_key)


def exa_search_companies(exa: Any, target_desc: str, offering_desc: str, limit: int = 5) -> List[Dict[str, str]]:
    """Use Exa to discover companies matching the targeting criteria.

    Returns list of {name, website, why_fit}.
    """
    query = (
        f"fashion retailer OR apparel brand site:*.com "
        f"AND (buying OR merchandising OR ecommerce) — {target_desc} — {offering_desc[:120]}"
    )
    try:
        results = exa.search_and_contents(query=query, num_results=limit, use_autoprompt=True)
        items = results.results if hasattr(results, "results") else results
    except Exception:
        items = exa.search(query=query, num_results=limit, use_autoprompt=True)
        items = items.results if hasattr(items, "results") else items

    companies: List[Dict[str, str]] = []
    for r in items:
        url = getattr(r, "url", None) or r.get("url", "")
        title = getattr(r, "title", None) or r.get("title", "")
        text = getattr(r, "text", None) or r.get("text", "")
        name = _infer_company_name(title, url)
        why = _summarize_fit_from_snippet(text or title, target_desc)
        companies.append({"name": name, "website": url, "why_fit": why})

    seen: set = set()
    unique: List[Dict[str, str]] = []
    for c in companies:
        domain = _domain(c.get("website", ""))
        if domain and domain not in seen:
            seen.add(domain)
            unique.append(c)
    return unique[:limit]


def exa_find_contacts(exa: Any, company: Dict[str, str]) -> List[Dict[str, Any]]:
    """Find 1–2 relevant contacts. Use role-based search and infer emails if needed."""
    domain = _domain(company.get("website", ""))
    name = company.get("name", "")
    roles = [
        "Head of Buying",
        "Merchandising Director",
        "VP Merchandising",
        "GTM Lead",
        "Partnerships Manager",
        "Product Marketing",
        "Founder",
    ]
    contacts: List[Dict[str, Any]] = []
    for role in roles:
        q = f"{role} {name} email OR contact"
        try:
            res = exa.search(query=q, num_results=2, use_autoprompt=True)
            items = res.results if hasattr(res, "results") else res
            for r in items:
                title = getattr(r, "title", None) or r.get("title", "")
                url = getattr(r, "url", None) or r.get("url", "")
                person = _infer_person_from_title(title)
                if person:
                    contacts.append({
                        "name": person,
                        "role": role,
                        "source": url,
                        "email": _infer_email(person, domain),
                        "inferred": True,
                    })
        except Exception:
            continue
        if len(contacts) >= 2:
            break
    return contacts[:2]


def exa_collect_research(exa: Any, company: Dict[str, str]) -> List[str]:
    """Collect 2–4 non-generic insights from website and Reddit mentions."""
    website = company.get("website", "")
    name = company.get("name", "")
    insights: List[str] = []
    queries = [
        f"site:{_domain(website)} about OR team OR blog",
        f"site:reddit.com {name}",
    ]
    for q in queries:
        try:
            res = exa.search_and_contents(query=q, num_results=3, use_autoprompt=True)
            items = res.results if hasattr(res, "results") else res
            for r in items:
                text = getattr(r, "text", None) or r.get("text", "")
                if not text:
                    continue
                for point in _extract_key_points(text):
                    insights.append(point)
        except Exception:
            continue
    seen: set = set()
    deduped: List[str] = []
    for s in insights:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            deduped.append(s.strip())
            if len(deduped) >= 4:
                break
    return deduped


def write_emails(
    openai_client: Any,
    email_style_key: str,
    sender_name: str,
    sender_company: str,
    calendar_link: Optional[str],
    companies: List[Dict[str, Any]],
    contacts_map: Dict[str, List[Dict[str, Any]]],
    research_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    style_instr = EMAIL_STYLES.get(email_style_key, EMAIL_STYLES["Professional"])
    outputs: List[Dict[str, Any]] = []
    for company in companies:
        company_name = company.get("name", "")
        for contact in contacts_map.get(company_name, []) or [{"name": "Team", "role": "Team"}]:
            prompt = _email_prompt(
                style_instr=style_instr,
                company=company,
                contact=contact,
                insights=research_map.get(company_name, []),
                sender_name=sender_name,
                sender_company=sender_company,
                calendar_link=calendar_link,
            )
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            text = resp.choices[0].message.content.strip()
            email = _parse_email_output(text)
            email.setdefault("company", company_name)
            email.setdefault("contact", contact.get("name", ""))
            outputs.append(email)
    return outputs


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Livetrend B2B Outreach (Demo)", layout="wide")

    st.title("Livetrend B2B Outreach — Multi-Agent Demo")
    st.caption("Inspired by Unwind AI tutorial; tailored for Livetrend use case.")

    st.sidebar.header("API Configuration")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    exa_key = st.sidebar.text_input("Exa API Key", type="password", value=os.getenv("EXA_API_KEY", ""))

    col1, col2 = st.columns(2)
    with col1:
        target_desc = st.text_area("Target companies", value=DEFAULT_TARGET_DESC, height=120)
        offering_desc = st.text_area("Your offering", value=DEFAULT_OFFERING_DESC, height=160)
    with col2:
        sender_name = st.text_input("Your name", value=os.getenv("SENDER_NAME", ""))
        sender_company = st.text_input("Your company", value=os.getenv("SENDER_COMPANY", "Livetrend"))
        num_companies = st.number_input("Number of companies", min_value=1, max_value=10, value=5)
        email_style = st.selectbox("Email style", list(EMAIL_STYLES.keys()), index=0)
        calendar_link = st.text_input("Calendar link (optional)", value=os.getenv("CALENDAR_LINK", ""))

    if st.button("Start Outreach", type="primary"):
        if not openai_key or not exa_key:
            st.error("Please provide both OpenAI and Exa API keys in the sidebar or .env")
            st.stop()

        try:
            clients = Clients(openai_key, exa_key)
        except Exception as e:
            st.error(str(e))
            st.stop()

        progress = st.progress(0)
        stage = st.empty()

        stage.info("1/4 Finding companies…")
        companies = exa_search_companies(clients.exa, target_desc, offering_desc, int(num_companies))
        progress.progress(25)
        st.subheader("Companies")
        st.write(companies)

        stage.info("2/4 Finding contacts…")
        contacts_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in companies:
            contacts_map[c["name"]] = exa_find_contacts(clients.exa, c)
        progress.progress(50)
        st.subheader("Contacts")
        st.write(contacts_map)

        stage.info("3/4 Gathering research…")
        research_map: Dict[str, List[str]] = {}
        for c in companies:
            research_map[c["name"]] = exa_collect_research(clients.exa, c)
        progress.progress(75)
        st.subheader("Research Insights")
        st.write(research_map)

        stage.info("4/4 Writing emails…")
        emails = write_emails(
            openai_client=clients.openai,
            email_style_key=email_style,
            sender_name=sender_name or "",
            sender_company=sender_company or "Livetrend",
            calendar_link=calendar_link or None,
            companies=companies,
            contacts_map=contacts_map,
            research_map=research_map,
        )
        progress.progress(100)
        stage.success("Completed")

        st.subheader("Generated Emails")
        for e in emails:
            with st.expander(f"{e.get('company', '')} — {e.get('contact', '')}"):
                st.write(e)


def _infer_company_name(title: str, url: str) -> str:
    if title:
        clean = re.sub(r"\s*[-|•].*$", "", title).strip()
        if len(clean.split()) <= 6:
            return clean
    d = _domain(url)
    return d.split(".")[0].capitalize() if d else url


def _domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url)
    return m.group(1) if m else url


def _summarize_fit_from_snippet(text: str, target_desc: str) -> str:
    text = (text or "").strip()
    return (
        "Matches targeting; retailer/brand with relevant assortment and ecommerce presence."
        if not text else text[:160]
    )


def _infer_person_from_title(title: str) -> Optional[str]:
    m = re.match(r"([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)", title or "")
    return m.group(1) if m else None


def _infer_email(person_name: str, domain: str) -> Optional[str]:
    if not person_name or not domain:
        return None
    parts = person_name.split(" ")
    first, last = parts[0], parts[-1]
    base = f"{first}.{last}".lower()
    clean_domain = domain.lower()
    return f"{base}@{clean_domain}"


def _extract_key_points(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    points: List[str] = []
    for s in sentences:
        ls = s.lower()
        if 40 <= len(s) <= 200 and ("launch" in ls or "growth" in ls or "strategy" in ls or "trend" in ls or "assort" in ls):
            points.append(s.strip())
    return points[:4]


def _email_prompt(
    style_instr: str,
    company: Dict[str, Any],
    contact: Dict[str, Any],
    insights: List[str],
    sender_name: str,
    sender_company: str,
    calendar_link: Optional[str],
) -> str:
    insights_text = "\n".join(f"- {i}" for i in (insights or [])) or "- [No strong public insights found]"
    contact_line = f"{contact.get('name','')} ({contact.get('role','')})".strip()
    cta = (
        f"If helpful, grab 15 min here: {calendar_link}" if calendar_link else "Open to a quick diagnostic chat next week?"
    )
    return (
        f"You are EmailWriterAgent. {style_instr}\n"
        f"Length: 120-160 words. Return JSON with keys: subject, body.\n"
        f"Company: {company.get('name','')} — {company.get('website','')}\n"
        f"Contact: {contact_line}\n"
        f"Why fit: {company.get('why_fit','')}\n"
        f"Insights:\n{insights_text}\n"
        f"Sender: {sender_name} at {sender_company}. CTA: {cta}\n"
    )


def _parse_email_output(text: str) -> Dict[str, str]:
    try:
        import json
        return json.loads(text)
    except Exception:
        subject_match = re.search(r"subject\s*:\s*(.*)", text, flags=re.IGNORECASE)
        body_start = text.find("body")
        return {
            "subject": subject_match.group(1).strip() if subject_match else "[No subject]",
            "body": text[body_start:].strip() if body_start != -1 else text,
        }


if __name__ == "__main__":
    main()
