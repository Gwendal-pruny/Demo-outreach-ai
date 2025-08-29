import os
import json
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from schemas import CompaniesPayload, ContactsPayload, ResearchPayload, EmailsPayload
from agents import (
	create_company_finder_agent,
	create_contact_finder_agent,
	create_research_agent,
	create_email_writer_agent,
	create_quality_check_agent,
)
from pipeline import (
	run_company_finder,
	run_contact_finder,
	run_research,
	run_email_writer,
	run_quality_check,
)
from utils import ensure_env, save_exports, payload_to_df, log_info


LIVETREND_OFFERING = (
	"Livetrend est une plateforme d'intelligence mode : analyses de tendances data-driven,"
	" veille concurrentielle, planification d'assortiment et optimisation pricing/achat. Nous collectons"
	" des données e-commerce, défilés et social pour aider marques/retailers à créer des best-sellers."
)

TARGET_EXAMPLE = (
	"Retailers mode ou marques DTC (EU/US) avec e-commerce actif, 50–1000 employés.\n"
	"Priorité : organisations pilotées par le merchandising (buying, planning), fast-fashion ou data-driven."
)


def main() -> None:
	load_dotenv()
	st.set_page_config(page_title="Outreach Livetrend", layout="wide")

	st.title("Outreach Livetrend — Demo (Agno)")
	st.caption("Mode Simple pour marketing, Mode Avancé pour power users. Affichage uniquement; aucun envoi.")

	st.sidebar.header("Clés API")
	openai_key = st.sidebar.text_input("Clé OpenAI", type="password", value=os.getenv("OPENAI_API_KEY", ""))
	exa_key = st.sidebar.text_input("Clé Exa", type="password", value=os.getenv("EXA_API_KEY", ""))

	mode = st.sidebar.radio("Mode", ["Simple", "Avancé"], index=0)

	st.sidebar.header("Modèles")
	model_options = ["gpt-5", "gpt-4o", "gpt-4o-mini"]
	if mode == "Avancé":
		company_model = st.sidebar.selectbox("Modèle Entreprises", model_options, index=0)
		contact_model = st.sidebar.selectbox("Modèle Contacts", model_options, index=1)
		research_model = st.sidebar.selectbox("Modèle Recherche", model_options, index=0)
		email_model = st.sidebar.selectbox("Modèle Emails", model_options, index=0)
		qc_enabled = st.sidebar.toggle("Contrôle qualité (QC)", value=True)
		followups_num = st.sidebar.slider("Suivis par contact", 0, 3, 1)
		openai_only = st.sidebar.toggle("OpenAI uniquement (sans fallback)", value=False, help="Désactive les fallbacks Exa/Template et retente OpenAI")
		retries = st.sidebar.slider("Relances OpenAI (si JSON vide)", 0, 3, 1)
		st.sidebar.header("Filtres")
		industries = st.sidebar.text_input("Industries (séparées par virgule)", value="fashion, apparel, retail").split(",")
		industries = [s.strip() for s in industries if s.strip()]
		regions = st.sidebar.text_input("Régions (séparées par virgule)", value="EU, US").split(",")
		regions = [s.strip() for s in regions if s.strip()]
		exclude_domains = st.sidebar.text_input("Exclure domaines (séparés par virgule)", value="livetrend.co").split(",")
		exclude_domains = [s.strip() for s in exclude_domains if s.strip()]
	else:
		global_model = st.sidebar.selectbox("Modèle (Simple)", model_options, index=2, help="Choisissez GPT-5 si disponible dans votre compte")
		company_model = contact_model = research_model = email_model = global_model
		qc_enabled = True
		followups_num = 1
		openai_only = False
		retries = 1
		industries = ["fashion", "apparel", "retail"]
		regions = ["EU", "US"]
		exclude_domains = ["livetrend.co"]

	col1, col2 = st.columns(2)
	with col1:
		target_desc = st.text_area("Cibles (exemples)", value=TARGET_EXAMPLE, height=120)
		offering_desc = st.text_area("Notre offre (pré-remplie)", value=LIVETREND_OFFERING, height=120)
	with col2:
		sender_name = st.text_input("Votre nom", value="")
		sender_company = st.text_input("Entreprise", value="Livetrend", disabled=True)
		num_companies = st.number_input("Nombre d'entreprises", min_value=1, max_value=10, value=5)
		email_style = st.selectbox("Style d'email", ["Professional", "Casual", "Cold", "Consultative"], index=0)

	st.caption(f"Modèles utilisés → Entreprises: {company_model}, Contacts: {contact_model}, Recherche: {research_model}, Emails: {email_model}")

	out_dir = "out"

	cta_label = "Lancer l'outreach" if mode == "Simple" else "Exécuter le pipeline"
	if st.button(cta_label, type="primary"):
		if not openai_key or not exa_key:
			st.error("Veuillez renseigner les clés API dans la barre latérale")
			st.stop()
		ensure_env(openai_key, exa_key)

		log_info(f"[CONFIG] Models → company:{company_model}, contact:{contact_model}, research:{research_model}, email:{email_model}; openai_only={openai_only}; retries={retries}")

		company_agent = create_company_finder_agent(company_model, industries, regions, exclude_domains)
		contact_agent = create_contact_finder_agent(contact_model)
		research_agent = create_research_agent(research_model)
		email_agent = create_email_writer_agent(email_model, email_style, followups=followups_num)
		qc_agent = create_quality_check_agent(email_model) if qc_enabled else None

		progress = st.progress(0)
		stage = st.empty()

		try:
			stage.info("1/5 Entreprises…")
			companies: CompaniesPayload = run_company_finder(company_agent, target_desc, offering_desc, int(num_companies), retries=retries, allow_fallbacks=not openai_only)
			progress.progress(20)
			st.subheader("Entreprises")
			st.dataframe(payload_to_df(companies))
			save_exports(out_dir, "companies", companies.model_dump())

			stage.info("2/5 Contacts…")
			contacts: ContactsPayload = run_contact_finder(contact_agent, companies, target_desc, offering_desc, retries=retries, allow_fallbacks=not openai_only)
			progress.progress(40)
			st.subheader("Contacts")
			st.dataframe(payload_to_df(contacts))
			save_exports(out_dir, "contacts", contacts.model_dump())

			stage.info("3/5 Recherche…")
			research: ResearchPayload = run_research(research_agent, companies, retries=retries, allow_fallbacks=not openai_only)
			progress.progress(60)
			st.subheader("Insights")
			st.dataframe(payload_to_df(research))
			save_exports(out_dir, "research", research.model_dump())

			stage.info("4/5 Emails…")
			sender = {"name": sender_name or "", "company": sender_company}
			emails: EmailsPayload = run_email_writer(email_agent, companies, contacts, research, sender=sender, style=email_style, retries=retries, allow_fallbacks=not openai_only)
			progress.progress(80)
			st.subheader("Emails générés")
			for e in emails.emails:
				with st.expander(f"{e.subject} — {e.company} / {e.contact}"):
					st.write(e.body)
					st.code(e.body)
			save_exports(out_dir, "emails", emails.model_dump())

			if qc_agent and emails.emails:
				stage.info("5/5 Contrôle qualité…")
				qc_results: List[Dict[str, Any]] = run_quality_check(qc_agent, emails)
				progress.progress(100)
				st.subheader("QC")
				st.json({"qc": qc_results})
				save_exports(out_dir, "qc", {"qc": qc_results})
			else:
				progress.progress(100)

			stage.success("Terminé")
		except Exception as e:
			stage.error("Échec du pipeline")
			st.error(f"Erreur: {str(e)}")


if __name__ == "__main__":
	main()
