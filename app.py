import os
import sys
import pdfplumber
import re
import json
import numpy as np
from src.llm_client import generate_llm_response
from src.embedding_model import get_embedding
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_required, current_user, logout_user, login_user
from bson.objectid import ObjectId
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
load_dotenv()
from auth.auth_routes import auth, User
from src.database import save_prediction, get_all_documents, users_collection, predictions_collection
from src.rag_engine import retrieve_relevant_laws
from werkzeug.security import check_password_hash, generate_password_hash
from src.ml_models import extract_case_name, extract_judge_name, extract_parties, extract_dates, extract_entities
from src.legal_analyzer import extract_case_citations
from src.law_domains import LAW_DOMAIN_MAP, KEYWORD_DOMAIN_MAP
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

app = Flask(
    __name__,
    template_folder="src/templates",
    static_folder="src/static"
)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret")

app.register_blueprint(auth)
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = None
login_manager.init_app(app)

def clean_mongo_data(data):
    """
    Recursively convert ObjectId and datetime
    so they become JSON serializable.
    """
    if isinstance(data, dict):
        return {key: clean_mongo_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_mongo_data(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.strftime("%Y-%m-%d %H:%M")
    else:
        return data
    
@app.context_processor
def inject_case_data():
    if not current_user.is_authenticated:
        return dict(caseData=[])
    docs = list(
        predictions_collection
        .find({"user_id": current_user.id})
        .sort("_id", -1)
    )
    docs = clean_mongo_data(docs)
    search_data = []
    for d in docs:
        if d.get("type") == "note":
            search_data.append({
                "_id": d["_id"],
                "type": "note",
                "title": d.get("case_name", "Note"),
                "main_category": "Note",
                "status": "Saved Note",
                "timestamp": ""
            })
        else:
            search_data.append({
                "_id": d["_id"],
                "type": "case",
                "title": d.get("case_name", "Case"),
                "main_category": d.get("main_category", ""),
                "status": d.get("status", "Case"),
                "timestamp": d.get("timestamp", "")
            })
    return dict(caseData=search_data)

#----- USER LOADER -----
@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(user)
    return None

def detect_document_type(text):
    text_lower = text.lower()
    if "press release" in text_lower or "stakeholders consultation" in text_lower:
        return "Press Release"

    if "an act to" in text_lower and "be it enacted" in text_lower:
        return "Statute"

    if "versus" in text_lower or " v " in text_lower or " vs " in text_lower:
        return "Judgment"

    if "petitioner" in text_lower and "respondent" in text_lower:
        return "Petition"

    if "agreement" in text_lower or "contract" in text_lower:
        return "Contract"

    if "order" in text_lower and "court" in text_lower:
        return "Court Order"

    return "Legal Document"

def classify_main_category(text):

    text_lower = text.lower()

    # -------- STEP 1: ACT BASED CLASSIFICATION --------
    detected_acts = detect_acts(text)

    priority_order = [
        "POCSO Act",
        "Information Technology Act",
        "Consumer Protection Act",
        "Indian Contract Act",
        "Transfer of Property Act",
        "Indian Penal Code"
    ]

    for act in priority_order:
        if act in detected_acts and act in LAW_DOMAIN_MAP:
            return LAW_DOMAIN_MAP[act]

    # -------- STEP 2: KEYWORD BASED CLASSIFICATION --------
    for domain, keywords in KEYWORD_DOMAIN_MAP.items():
        for word in keywords:
            if word in text_lower:
                return domain

    # -------- STEP 3: AI CLASSIFICATION --------
    prompt = f"""
    You are an expert legal document classifier.

    Classify the document into the most appropriate legal domain.

    Possible domains:
    Criminal Law
    Civil Law
    Constitutional Law
    Administrative Law
    Corporate Law
    Labour Law
    Intellectual Property Law
    Tax Law
    Family Law
    Child Protection (POCSO)
    Cyber Law
    Consumer Protection
    Environmental Law
    Property Law
    Banking Law
    Insurance Law
    Arbitration Law
    Contract Law
    Company Law
    Competition Law

    Return ONLY the domain name.

    Document:
    {text[:3000]}
    """

    try:
        category = generate_llm_response(prompt).strip()

        if not category:
            return "General Law"

        print("Domain detected by LLM:", category)

        return category

    except Exception as e:
        print("Classification Error:", e)
        return "General Law"

def extract_legal_sections(text):

    sections = []
    patterns = [
    r"section\s+(\d{1,3}[a-z]?(?:\(\d+\))?)\b",
    r"sections?\s+(\d{1,3})\s*(?:and|,)\s*(\d{1,3})",
    r"u/s\s+(\d{1,3})",
    r"\b(\d{1,3})\s*ipc\b",
    r"\b(\d{1,3})\s*crpc\b",
    r"\b(\d{1,3})\s*cpc\b"
    ]
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                for m in match:
                    sections.append({
                        "section": f"Section {m}",
                        "title": f"Section {m}"
                    })
            else:
                sections.append({
                    "section": f"Section {match}",
                    "title": f"Section {match}"
                })

    sections = list({s["section"]: s for s in sections}.values())
    sections = sorted(sections, key=lambda x: x["section"])

    return sections[:5]

def build_subdivisions_from_sections(sections):
    subdivisions = []
    for s in sections:
        title = s["title"].lower()
        if "assault" in title:
            severity = "High"
        elif "punishment" in title:
            severity = "High"
        elif "procedure" in title:
            severity = "Medium"
        elif "definition" in title:
            severity = "Low"
        else:
            severity = "Medium"
        subdivisions.append({
            "title": f"Legal Provision: {s['section']}",
            "law": s["section"],
            "severity": severity,
            "explanation": f"This provision explains {s['title']} under the Act."
        })
    return subdivisions

def detect_acts(text):
    acts = []
    act_patterns = {
        "Indian Penal Code": r"\b(IPC|I\.P\.C\.|Indian Penal Code)\b",
        "Code of Criminal Procedure": r"\b(CrPC|Cr\.P\.C\.|Code of Criminal Procedure)\b",
        "Code of Civil Procedure": r"\b(CPC|C\.P\.C\.|Code of Civil Procedure)\b",
        "POCSO Act": r"\b(POCSO|Protection of Children from Sexual Offences Act)\b",
        "Information Technology Act": r"\b(IT Act|Information Technology Act|IT Act 2000)\b",
        "Indian Contract Act": r"\b(Contract Act|Indian Contract Act)\b",
        "Transfer of Property Act": r"\bTransfer of Property Act\b",
        "Domestic Violence Act": r"\b(DV Act|Domestic Violence Act)\b",
        "Hindu Marriage Act": r"\bHindu Marriage Act\b",
        "Consumer Protection Act": r"\bConsumer Protection Act\b",
        "Negotiable Instruments Act": r"\bNegotiable Instruments Act\b",
        "Companies Act": r"\bCompanies Act\b",
        "Insolvency and Bankruptcy Code": r"\b(IBC|Insolvency and Bankruptcy Code)\b",
        "NDPS Act": r"\bNDPS\b",
        "Motor Vehicles Act": r"\bMotor Vehicles Act\b"
    }
    for act, pattern in act_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            acts.append(act)

    return list(set(acts))

def extract_court_decision(text):
    patterns = [
        r"sentence\s+(?:shall\s+be\s+)?suspended",
        r"appeal\s+(?:is\s+)?allowed",
        r"appeal\s+(?:is\s+)?dismissed",
        r"petition\s+(?:is\s+)?allowed",
        r"petition\s+(?:is\s+)?dismissed",
        r"application\s+(?:is\s+)?allowed",
        r"application\s+(?:is\s+)?dismissed",
        r"bail\s+(?:is\s+)?granted",
        r"bail\s+(?:is\s+)?rejected"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group().capitalize()
    return "Decision not clearly identified"

def generate_case_summary(text):

    text_lower = text.lower()

    # ----- STATUTE SUMMARY -----
    if "be it enacted" in text_lower or "an act to" in text_lower:

        prompt = f"""
        Summarize the following Indian legal statute.

        Include:
        • Purpose of the Act
        • Key provisions
        • What offences it regulates
        • Why the law exists

        Keep summary between 80–120 words.

        Document:
        {text[:3000]}
        """

    # ----- COURT CASE SUMMARY -----
    else:

        prompt = f"""
        Summarize the following Indian legal case.

        Include:
        • Background of the case
        • Key legal issue
        • Relevant law
        • Court reasoning

        Rules:
        - Keep summary between 80–120 words
        - Use only information from the document

        Case:
        {text[:3000]}
        """

    try:
        summary = generate_llm_response(prompt) or "Summary unavailable."
        summary = summary.strip()
        summary = re.sub(r"[*#]", "", summary)
        summary = re.sub(r"\s+", " ", summary)
        return summary

    except Exception as e:
        print("Summary error:", e)
        return "Summary could not be generated."

def clean_sections(section_list):
    cleaned = []
    for s in section_list:
        s = s.strip()
        if "read with" in s.lower():
            parts = re.findall(r"Section\s+\d+[A-Za-z]*", s, re.IGNORECASE)
            cleaned.extend(parts)
        else:
            cleaned.append(s)
    final = []
    for sec in cleaned:
        sec = re.sub(r"Sections?", "Section", sec, flags=re.IGNORECASE)
        sec = re.sub(r"\s+", " ", sec)
        final.append(sec.strip())
    return sorted(list(set(final)))

def normalize_sections(sections):

    normalized = set()

    for s in sections:
        s = s.lower()

        match = re.search(r"section\s+(\d+[a-z]*)", s)
        if not match:
            continue

        number = match.group(1)

        if "ipc" in s:
            normalized.add(f"Section {number} IPC")
        elif "crpc" in s:
            normalized.add(f"Section {number} CrPC")
        else:
            normalized.add(f"Section {number}")

    return sorted(list(normalized))

def explain_sections_with_ai(section_list, text, acts):
    explanations = []
    for sec in section_list[:5]:
        if isinstance(sec, dict):
            sec = sec.get("section")
        prompt = f"""
        Explain the Indian legal provision {sec}.
        Possible Acts mentioned in the document:
        {acts}
        If the section belongs to one of these Acts, explain it.
        If not, explain it under the most relevant Indian law.
        Provide:
        1. Act name
        2. What the section defines
        3. Punishment if applicable
        4. Example
        IMPORTANT:
        Explain the section ONLY under the detected Act above.
        Do not assume another Act.
        Keep explanation short.
        Document context:
        {text[:1200]}
        """
        try:
            explanation = generate_llm_response(prompt).strip()
            if not explanation or len(explanation) < 10:
                explanation = f"{sec} is a legal provision under Indian law."
        except:
            explanation = f"{sec} is a legal provision under Indian law."
        explanations.append({
            "section": sec,
            "explanation": explanation
        })
    return explanations

DOMAIN_SUBDIVISIONS = {

"Criminal Law": [
{
"title":"Cheating and Fraud",
"explanation":"Cases involving deception, financial fraud, or dishonest inducement of property under provisions like Section 420 IPC."
},
{
"title":"Forgery and Fake Documents",
"explanation":"Cases involving creation or use of forged documents under provisions such as Sections 465–471 IPC."
},
{
"title":"Criminal Conspiracy",
"explanation":"Situations where multiple persons agree to commit an illegal act, punishable under Section 120B IPC."
},
{
"title":"Bail Applications",
"explanation":"Legal proceedings where an accused person seeks temporary release from custody pending investigation or trial."
},

{
"title":"Suspension of Sentence",
"explanation":"Applications filed before appellate courts requesting suspension of conviction or imprisonment during appeal."
},

{
"title":"Assault and Violence",
"explanation":"Cases involving physical harm, intimidation, or violent offences including assault and grievous hurt."
}

],

"Property Law":[

{
"title":"Partition of Property",
"explanation":"Disputes among family members or co-owners regarding division and separate possession of property."
},

{
"title":"Specific Performance of Agreement",
"explanation":"Cases seeking enforcement of agreements for sale of property under the Specific Relief Act."
},

{
"title":"Sale Deed Validity",
"explanation":"Disputes questioning whether a sale deed was legally executed or validly transferred ownership."
},

{
"title":"Ownership and Title Disputes",
"explanation":"Cases determining the lawful ownership or title of land or immovable property."
},

{
"title":"Transfer of Property",
"explanation":"Issues relating to legal transfer of ownership under the Transfer of Property Act."
},

{
"title":"Injunction Orders",
"explanation":"Court orders preventing parties from selling, altering, or interfering with property rights."
}

],

"Civil Law":[

{
"title":"Contract Disputes",
"explanation":"Cases involving breach or enforcement of agreements between parties."
},

{
"title":"Property Ownership Disputes",
"explanation":"Civil disputes determining lawful ownership or possession of land or property."
},

{
"title":"Recovery of Money",
"explanation":"Cases filed to recover unpaid debts, loans, or contractual payments."
},

{
"title":"Injunction Proceedings",
"explanation":"Civil remedies where courts order a party to stop or perform a particular act."
},

{
"title":"Specific Performance",
"explanation":"Cases seeking court enforcement of contractual obligations."
}

],

"Child Protection (POCSO)": [

{
"title":"Penetrative Sexual Assault",
"explanation":"Offences defined under Sections 3 and 4 of the POCSO Act involving sexual assault against minors."
},

{
"title":"Aggravated Sexual Assault",
"explanation":"Serious offences involving abuse of authority, multiple offenders, or severe harm to a child."
},

{
"title":"Sexual Harassment of Child",
"explanation":"Acts involving inappropriate touching or harassment under Sections 11 and 12 of the POCSO Act."
},

{
"title":"Child Pornography",
"explanation":"Offences related to using minors for pornographic purposes under Sections 13–15."
},

{
"title":"Trial in Special Court",
"explanation":"Procedures for child-friendly trial conducted by Special Courts under Sections 28–38."
}

]

}

DOMAIN_SUBDIVISIONS.update({
    
"Banking Law":[
{"title":"Loan Default","explanation":"Failure of a borrower to repay a loan according to agreed terms."},
{"title":"Cheque Bounce","explanation":"Dishonour of cheque under provisions like Section 138 of the Negotiable Instruments Act."},
{"title":"Bank Fraud","explanation":"Fraudulent transactions involving banking institutions."},
{"title":"Recovery Proceedings","explanation":"Legal proceedings initiated by banks for loan recovery."}
],

"Family Law":[
{"title":"Divorce","explanation":"Legal proceedings seeking dissolution of marriage."},
{"title":"Child Custody","explanation":"Disputes over guardianship and custody of children."},
{"title":"Maintenance","explanation":"Claims for financial support between spouses."},
{"title":"Adoption","explanation":"Legal adoption of a child."},
{"title":"Inheritance","explanation":"Disputes relating to succession of property."}
],

"Cyber Law":[
{"title":"Online Fraud","explanation":"Fraud committed using digital platforms."},
{"title":"Identity Theft","explanation":"Unauthorized use of personal identity information."},
{"title":"Data Breach","explanation":"Unauthorized access to confidential digital data."},
{"title":"Cyber Harassment","explanation":"Harassment conducted through electronic communication."}
],

"Corporate Law":[
{"title":"Corporate Fraud","explanation":"Fraudulent activities conducted by company officials."},
{"title":"Shareholder Disputes","explanation":"Disputes between company shareholders."},
{"title":"Company Mismanagement","explanation":"Improper management of company affairs."},
{"title":"Corporate Insolvency","explanation":"Proceedings under insolvency and bankruptcy law."}
],

"Consumer Protection":[
{"title":"Defective Product","explanation":"Consumer complaints regarding faulty products."},
{"title":"Service Deficiency","explanation":"Failure to provide promised services."},
{"title":"Medical Negligence","explanation":"Negligence by medical professionals causing harm."}
],

"Labour Law":[
{"title":"Wrongful Termination","explanation":"Illegal termination of employment."},
{"title":"Wage Disputes","explanation":"Disputes related to payment of wages."},
{"title":"Industrial Disputes","explanation":"Disputes between workers and employers."}
]

})

SECTION_SUBDIVISION_MAP = {

"120B": {
"title": "Criminal Conspiracy",
"explanation": "Agreement between two or more persons to commit an illegal act under Section 120B IPC."
},

"406": {
"title": "Criminal Breach of Trust",
"explanation": "Misappropriation of property entrusted to a person under Section 406 IPC."
},

"409": {
"title": "Breach of Trust by Public Servant",
"explanation": "Serious offence where a public servant dishonestly misappropriates entrusted property."
},

"420": {
"title": "Cheating and Fraud",
"explanation": "Dishonest inducement causing delivery of property under Section 420 IPC."
},

"430": {
"title": "Mischief Causing Damage",
"explanation": "Intentional damage to property or public infrastructure."
},

"3": {
"title": "Penetrative Sexual Assault",
"explanation": "Offence involving sexual penetration against a minor under Section 3 of the POCSO Act."
},

"4": {
"title": "Punishment for Penetrative Sexual Assault",
"explanation": "Punishment for penetrative sexual assault under Section 4 of the POCSO Act."
},

"7": {
"title": "Sexual Assault on Child",
"explanation": "Sexual assault without penetration under Section 7 of the POCSO Act."
},

"8": {
"title": "Punishment for Sexual Assault",
"explanation": "Punishment prescribed under Section 8 of the POCSO Act."
}

}

def generate_subdivisions_from_sections(sections):
    subdivisions = []
    for sec in sections:
        number = re.findall(r"\d+", sec)
        if not number:
            continue
        number = number[0]
        if number in SECTION_SUBDIVISION_MAP:
            info = SECTION_SUBDIVISION_MAP[number]
            subdivisions.append({
                "title": info["title"],
                "law": f"Section {number} IPC",
                "severity": "High",
                "explanation": info["explanation"]
            })
    return subdivisions

#----- RAG + LLM -----
def analyze_subdivisions_llm(text, main_category):
    detected_sections = [s["section"] for s in extract_legal_sections(text)]
    retrieved_laws = retrieve_relevant_laws(text, k=5)
    law_context = "\n".join(
        [f"{law['title']}: {law['content'][:500]}" for law in retrieved_laws]
    )
    prompt = f"""
    You are a senior Indian legal analyst specializing in court judgment analysis.
    STRICT RULES:
    1. Extract legal provisions ONLY if the exact text appears in the document.
    2. Do NOT infer or guess any legal section.
    3. If a section is not literally written in the text, do not include it.
    4. If none appear, return an empty list.
    "law": "Not specified in document"
    Never use generic titles like "Legal Issue".
    Always generate a meaningful issue name based on the case.
    ANALYSIS REQUIREMENTS:
    • Identify meaningful legal issues instead of generic titles.
    • Avoid repeating "Legal Issue".
    • Combine related sections where appropriate (e.g., Section 420 read with Section 120B IPC).
    • Provide concise but professional legal explanations.
    • Base reasoning strictly on the provided document.
    Also extract the FINAL COURT DECISION if present.
    Example:
    Sentence suspended
    Appeal allowed
    Bail granted
    Detected Sections in Document:
    {detected_sections}
    Main Category: {main_category}
    Relevant Legal Provisions Retrieved:
    {law_context}
    Return STRICT JSON.
    Return ONLY raw JSON.
    Do not include markdown formatting.
    Do not include explanations.
    {{
    "main_category": "{main_category}",
    "nature_of_dispute": "Clear summary of the dispute from the document",
    "legal_provisions": ["List only provisions explicitly present in the document"],
    "subdivisions": [
        {{
        "title": "Short descriptive title of the issue based on document (e.g., Exemption from Personal Appearance, Adjournment of Case, Suspension of Sentence). NEVER use 'Legal Issue'.",
        "law": "Exact section written in document OR Not specified in document",
        "severity": "Low/Medium/High",
        "explanation": "Clear legal reasoning explaining the issue and its relevance"
        }}
    ],
    "confidence": "integer between 0 and 100"
    }}
    Case Document:
    {text[:4000]}
    """
    try:
        output = generate_llm_response(prompt)
        print("LLM OUTPUT:", output)
        match = re.search(r"\{[\s\S]*\}", output)
        if match:
            analysis = json.loads(match.group().strip())
        else:
            raise ValueError("Invalid JSON returned by LLM")

        all_sections = analysis.get("legal_provisions", []) + detected_sections
        cleaned = clean_sections(all_sections)
        analysis["legal_provisions"] = list(set(normalize_sections(cleaned)))
        GENERIC_TITLES = [
            "legal issue",
            "legal matter",
            "legal provision",
            "legal case",
            "legal dispute"
        ]

        for sub in analysis.get("subdivisions", []):
            title = sub.get("title", "").lower()

            if title in GENERIC_TITLES or len(title) < 5:
                law = sub.get("law", "")

                if "pocso" in law.lower():
                    sub["title"] = "Sexual Offence Against Minor"
                elif "420" in law:
                    sub["title"] = "Cheating and Fraud"
                elif "120b" in law:
                    sub["title"] = "Criminal Conspiracy"
                elif "13" in law and "marriage" in law.lower():
                    sub["title"] = "Divorce Petition"
                else:
                    sub["title"] = f"{main_category} Legal Issue"
        return analysis
    except Exception as e:
        print("RAG Analysis Error:", e)
        return {
            "main_category": main_category,
            "nature_of_dispute": "The document appears to contain a legal dispute or statutory provision.",
            "legal_provisions": detected_sections[:5],
            "subdivisions": [{
                "title": f"General {main_category} Matter",
                "law": "Not specified in document",
                "severity": "Medium",
                "explanation": "The system could not extract structured legal issues, but the document relates to a legal matter."
            }],
            "confidence": 75
        }

#----- LLM Explanation -----
def generate_llm_explanation(text, main_category, subdivisions, confidence):
    subdivision_text = "\n".join(
        [f"- {s['title']} ({s['law']})" for s in subdivisions]
    ) if subdivisions else "No specific subdivisions detected."
    prompt = f"""
    You are a senior Indian legal analyst.
    Analyze the case document and explain:
    1. Why this case belongs to the identified legal domain.
    2. What legal provisions are involved.
    3. What the court decision indicates.
    4. The overall legal reasoning of the judgment.
    Detected Category: {main_category}
    Detected Legal Issues:
    {subdivision_text}
    Document:
    {text[:3000]}
    Provide a professional legal explanation.
    """
    try:
        return generate_llm_response(prompt)
    except:
        return "AI explanation could not be generated."
    
#----- Home Route -----
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():

    if request.method == "GET":
        return render_template("analyze.html")
    
    text = ""
    source = "Manual Text Entry"
    if request.form.get("judgement"):
        text = request.form["judgement"]
    elif request.files.get("pdf_file"):
        pdf = request.files["pdf_file"]
        source = "PDF Upload"
        with pdfplumber.open(pdf) as pdf_doc:
            pages = [page.extract_text() or "" for page in pdf_doc.pages]
            text = "\n".join(pages)[:10000]
    if not text.strip():
        return redirect(url_for("analyze"))

    doc_type = detect_document_type(text)
    if doc_type == "Statute":
        case_name = "Statutory Legal Document"
        petitioner = None
        respondent = None

    main_category = classify_main_category(text)
    if doc_type == "Statute":
        sections = extract_legal_sections(text)
        subdivisions = build_subdivisions_from_sections(sections)
        analysis = {
            "main_category": main_category,
            "nature_of_dispute": "Statutory legal framework.",
            "legal_provisions": [s["section"] for s in sections],
            "subdivisions": subdivisions,
            "confidence": 95
        }
    else:
        analysis = analyze_subdivisions_llm(text, main_category)

    if doc_type != "Statute":

        case_name = extract_case_name(text)
        if "appeal no" in case_name.lower():
            lines = text.split("\n")
            for line in lines[:40]:
                if " v " in line.lower() or " vs " in line.lower():
                    case_name = line.strip()
                    break
        case_name = re.sub(r"\(arising.*?\)", "", case_name, flags=re.I)
        case_name = case_name.strip()
        case_name = re.sub(r"Civil Appeal No.*", "", case_name, flags=re.I)
        case_name = re.sub(r"SLP.*No.*", "", case_name, flags=re.I)
        case_name = case_name.strip()
        petitioner, respondent = extract_parties(case_name)
        if case_name == "Unknown Case":
            lines = text.split("\n")
            for line in lines[:30]:
                line_clean = line.strip()
                if len(line_clean) < 10:
                    continue

                if " v " in line_clean.lower() or " vs " in line_clean.lower():
                    if "arising out of" in line_clean.lower():
                        continue

                    if "slp" in line_clean.lower():
                        continue
                    case_name = line_clean[:120]
                    break

    court = detect_court(text)
    judge = extract_judge_name(text)
    citations = extract_case_citations(text)
    dates = extract_dates(text)
    timeline = extract_case_timeline(text)
    amounts = extract_monetary_amounts(text)
    entities = extract_entities(text) or {}
    if entities:
        blacklist = [
            "schedule","gazette","miscellaneous","section","sec",
            "court","case","act","law","order",
            "assault","child","clause","rule","subsection",
            "article","chapter","part","date","year",
            "ors","and ors","bhartiya","page","crl","c.a.",
            "slp","lrs","ors","anr","respondent","appellant",
            "civil judge","high court","supreme court"
        ]
        blacklist += [
            "leave","leave granted","leave to appeal","j.","justice",
            "inter alia","designation","chapter","special courts"
        ]
        cleaned_persons = []
        for p in entities.get("persons", []):
            p_clean = p.strip()
            if len(p_clean) < 3:
                continue
            if p_clean.lower() in blacklist:
                continue
            if p_clean.lower().startswith("section"):
                continue
            if " v. " in p_clean.lower():
                continue
            if any(char.isdigit() for char in p_clean):
                continue
            if "hobli" in p_clean.lower():
                continue
            if len(p_clean.split()) > 4:
                continue
            if p_clean.isupper():
                continue
            if "." in p_clean and len(p_clean) < 5:
                continue
            if "—" in p_clean:
                continue
            if "(" in p_clean:
                continue

            cleaned_persons.append(p_clean)

        cleaned_orgs = []
        for o in entities.get("organizations", []):
            o_clean = o.strip()
            if len(o_clean) < 3:
                continue
            if o_clean.lower() in blacklist:
                continue
            if o_clean.lower().startswith("section"):
                continue
            if "page" in o_clean.lower():
                continue
            if o_clean.lower() in ["ipc", "crpc", "cpc"]:
                continue
            if re.search(r"page\s*\d+", o_clean.lower()):
                continue

            cleaned_orgs.append(o_clean)

        entities["persons"] = list(set(cleaned_persons))
        entities["organizations"] = list(set(cleaned_orgs))
        entities.pop("locations", None)
    sections = normalize_sections(
        clean_sections(
            analysis.get("legal_provisions", []) +
            [s["section"] for s in extract_legal_sections(text)]
        )
    )
    subdivisions = analysis.get("subdivisions", [])
    if not subdivisions:
        subdivisions = generate_subdivisions_from_sections(sections)
    if not subdivisions:
        domain_issues = DOMAIN_SUBDIVISIONS.get(main_category, [])
        subdivisions = []
        for issue in domain_issues[:4]:
            subdivisions.append({
                "title": issue["title"],
                "law": "Based on case context",
                "severity": "Medium",
                "explanation": issue["explanation"]
            })
    if not subdivisions:
        subdivisions = [{
            "title": f"General {main_category} Matter",
            "law": "Not specified in document",
            "severity": "Medium",
            "explanation": "No explicit statutory provisions mentioned in the document."
        }]

    detected_acts = detect_acts(text)
    if doc_type == "Statute":
        court_decision = "Not applicable (statutory document)"
    else:
        court_decision = predict_case_outcome(text)

    outcome_confidence = "High" if court_decision != "Outcome Not Detected" else "Low"
    summary = generate_case_summary(text)
    if doc_type == "Statute":
        risk_analysis = "Risk analysis is not applicable for statutory legal documents."
    else:
        risk_analysis = predict_legal_risk(text, main_category)

    embedding_vector = get_embedding(text)
    embedding = embedding_vector.tolist() if embedding_vector is not None else []
    similar_cases = find_similar_cases(embedding, main_category)
    if doc_type == "Statute":
        legal_arguments = "Legal arguments are not applicable for statutory documents."
    else:
        legal_arguments = generate_legal_arguments(text, main_category)
    confidence_raw = str(analysis.get("confidence", "90"))
    match = re.search(r"\d+", confidence_raw)
    confidence = int(match.group()) if match else 90
    nature = analysis.get("nature_of_dispute", "")

    save_prediction(
        text=text,
        case_name=case_name,
        court=court,
        judge=judge,
        doc_type=doc_type,
        acts=detected_acts,
        main_category=main_category,
        nature_of_dispute=nature,
        source=source,
        confidence=confidence,
        subdivisions=subdivisions,
        user_id=current_user.id,
        court_decision=court_decision,
        summary=summary,
        embedding=embedding
    )

    section_explanations = explain_sections_with_ai(
        sections,
        text,
        detected_acts
    )

    return render_template(
        "analysis_result.html",
        main_category=analysis.get("main_category", main_category),
        case_name=case_name,
        petitioner=petitioner,
        respondent=respondent,
        citations=citations,
        dates=dates,
        timeline=timeline,
        amounts=amounts,
        entities=entities,
        subdivisions=subdivisions,
        nature=nature,
        legal_provisions=analysis.get("legal_provisions", []),
        section_explanations=section_explanations,
        confidence=confidence,
        source=source,
        court_decision=court_decision,
        summary=summary,
        risk_analysis=risk_analysis,
        legal_arguments=legal_arguments,
        acts=detected_acts,
        outcome_confidence=outcome_confidence,
        similar_cases=similar_cases
    )

@app.route("/ask_case_question", methods=["POST"])
@login_required
def ask_case_question():

    question = request.form.get("question")
    case_text = request.form.get("case_text")
    if not question or not case_text:
        return jsonify({"answer": "Question or case text missing."})

    chunks = [case_text[i:i+800] for i in range(0, len(case_text), 800)]
    context = "\n\n".join(chunks[:3])
    prompt = f"""
    You are a legal assistant.
    Case Context:
    {context}
    User Question:
    {question}
    Answer using ONLY the case context above.
    If the information is not mentioned, say:
    "The information is not mentioned in the case."
    """
    answer = generate_llm_response(prompt)

    return jsonify({"answer": answer})

@app.route("/search_cases", methods=["POST"])
@login_required
def search_cases():

    query = request.form.get("query")

    if not query:
        return redirect(url_for("dashboard"))

    query_embedding_vector = get_embedding(query)
    query_embedding = query_embedding_vector.tolist() if query_embedding_vector is not None else []
    similar_cases = find_similar_cases(query_embedding, "General Law")

    return render_template(
        "search_results.html",
        query=query,
        results=similar_cases
    )

def predict_legal_risk(text, main_category):
    prompt = f"""
    Evaluate the legal risk of the following case.
    Provide:
    Risk Level: Low / Medium / High
    Strengths:
    - bullet points
    Weaknesses:
    - bullet points
    Probability of Success: percentage
    Keep answer under 120 words.
    Case Category: {main_category}
    {text[:3000]}
    """
    try:
        return generate_llm_response(prompt)
    except:
        return "Risk analysis could not be generated."
    
def generate_legal_arguments(text, main_category):
    prompt = f"""
    You are a senior Indian lawyer.
    Based on the following case document generate:
    Petitioner Arguments:
    - Provide 3 strong legal arguments.
    Respondent Arguments:
    - Provide 3 strong legal arguments.
    Use bullet points.
    Case Category: {main_category}
    Document:
    {text[:3000]}
    """
    try:
        return generate_llm_response(prompt)
    except:
        return "Legal arguments could not be generated."

def extract_case_timeline(text):
    timeline = {}
    patterns = {
        "Incident Date": r"(incident|offence|crime)\s+(?:occurred\s+on|dated)\s+([A-Za-z0-9,\-\s]+)",
        "FIR Date": r"(fir\s+(?:registered|lodged)\s+on)\s+([A-Za-z0-9,\-\s]+)",
        "Charge Sheet Filed": r"(charge\s+sheet\s+(?:filed|submitted)\s+on)\s+([A-Za-z0-9,\-\s]+)",
        "Trial Court Judgment": r"(trial\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)",
        "High Court Decision": r"(high\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)",
        "Supreme Court Decision": r"(supreme\s+court\s+(?:judgment|order)\s+dated)\s+([A-Za-z0-9,\-\s]+)"
    }
    text_lower = text.lower()
    for label, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            timeline[label] = match.group(2).strip()

    return timeline

def extract_monetary_amounts(text):

    patterns = [
        r"₹\s?\d{1,3}(?:,\d{3})+",
        r"₹\s?\d+",
        r"rs\.?\s?\d+",
        r"\d+\s?rupees"
    ]
    amounts = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        amounts.extend(matches)
    cleaned = []
    for amt in amounts:
        amt = amt.replace("rs.", "₹").replace("rs", "₹")
        amt = amt.strip()
        cleaned.append(amt)

    return list(set(cleaned))[:3]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_cases(query_embedding, category, top_k=5):

    all_docs = list(
        predictions_collection.find(
            {
                "embedding": {"$exists": True},
                "main_category": category
            },
            {"embedding": 1, "case_name": 1, "main_category": 1}
        ).limit(500)
    )
    scores = []
    for doc in all_docs:
        emb = doc.get("embedding")
        if not emb or not query_embedding:
            continue
        if len(emb) != len(query_embedding):
            continue
        score = cosine_similarity(query_embedding, emb)
        if score >= 0.98:
            continue
        if score < 0.50:
            continue
        doc["similarity"] = round(score * 100, 2)
        scores.append((score, doc))
    scores.sort(reverse=True, key=lambda x: x[0])

    seen = set()
    results = []
    for score, doc in scores:
        name = doc.get("case_name")
        if name in seen:
            continue
        seen.add(name)
        results.append(doc)
        if len(results) >= top_k:
            break

    return results

OUTCOME_PATTERNS = {

"Sentence Suspended": r"sentence\s+(?:shall\s+be\s+)?suspended",
"Bail Granted": r"bail\s+(?:is\s+)?granted",
"Bail Rejected": r"bail\s+(?:is\s+)?rejected",
"Appeal Allowed": r"appeal\s+(?:is\s+)?allowed|appeal\s+succeeds",
"Appeal Dismissed": r"appeal\s+(?:is\s+)?dismissed|appeal\s+fails",
"Petition Allowed": r"petition\s+(?:is\s+)?allowed",
"Petition Dismissed": r"petition\s+(?:is\s+)?dismissed",
"Conviction Set Aside": r"conviction\s+(?:is\s+)?set\s+aside",
"Conviction Upheld": r"conviction\s+(?:is\s+)?upheld",
"Case Remanded": r"matter\s+is\s+remanded|case\s+is\s+remanded",
"Judgment Set Aside": r"judgment\s+.*set\s+aside",
"Decree Set Aside": r"decree\s+.*set\s+aside",
"Complaint Allowed": r"complaint\s+(?:is\s+)?allowed",
"Complaint Dismissed": r"complaint\s+(?:is\s+)?dismissed"

}

def predict_case_outcome(text):
    text_lower = text.lower()
    for outcome, pattern in OUTCOME_PATTERNS.items():
        if re.search(pattern, text_lower):
            return outcome

    return "Outcome Not Detected"

def detect_court(text):

    patterns = [
        r"supreme court of india",
        r"high court of [a-z\s]+",
        r"district court",
        r"session[s]? court",
        r"court of the civil judge"
    ]
    text_lower = text.lower()
    for p in patterns:
        match = re.search(p, text_lower)
        if match:
            return match.group().title()

    return "Court Not Identified"

@app.route("/cases")
@login_required
def my_cases():
    page = request.args.get("page", 1, type=int)
    per_page = 5
    total_docs = predictions_collection.count_documents({
        "user_id": current_user.id,
        "type": "note",
        "show_in_cases": {"$ne": False}
    })

    documents = list(
    predictions_collection
    .find({
        "user_id": current_user.id,
        "type": "note",
        "show_in_cases": {"$ne": False}
    })
        .sort("_id", -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    documents = clean_mongo_data(documents)
    total_pages = (total_docs + per_page - 1) // per_page

    return render_template(
        "cases.html",
        documents=documents,
        current_page=page,
        total_pages=total_pages
    )

@app.route("/save_note", methods=["POST"])
@login_required
def save_note():
    try:
        data = request.get_json()
        note = data.get("note")
        case_name = data.get("case_name")
        if not note or not case_name:
            return jsonify({"status": "error", "message": "missing data"})

        result = predictions_collection.insert_one({
            "user_id": current_user.id,
            "case_name": case_name,
            "note": note,
            "type": "note",
            "created_at": datetime.now()
        })
        return jsonify({
            "status": "success",
            "id": str(result.inserted_id)
        })

    except Exception as e:
        print("SAVE NOTE ERROR:", e)
        return jsonify({"status": "error"})

@app.route("/view_note/<id>")
@login_required
def view_note(id):
    doc = predictions_collection.find_one({
        "_id": ObjectId(id),
        "type": "note"
    })
    return render_template("view_note.html", doc=doc)

@app.route("/remove_case/<id>", methods=["POST"])
@login_required
def remove_case(id):
    predictions_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"show_in_cases": False}}
    )
    return redirect(url_for("my_cases"))

@app.route("/bookmarks")
@login_required
def bookmarks():
    documents = list(
        predictions_collection
        .find({
            "user_id": current_user.id,
            "bookmarked": True
        })
        .sort("_id", -1)
    )
    documents = clean_mongo_data(documents)
    return render_template(
        "bookmark.html",
        documents=documents
    )

@app.route("/bookmark/<id>", methods=["POST"])
@login_required
def bookmark_case(id):
    doc = predictions_collection.find_one({
        "_id": ObjectId(id),
        "user_id": current_user.id
    })
    if not doc:
        return jsonify({
            "status": "error",
            "message": "Document not found"
        })
    new_value = not doc.get("bookmarked", False)
    predictions_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"bookmarked": new_value}}
    )
    return jsonify({
        "status": "success",
        "bookmarked": new_value
    })

@app.route("/help")
@login_required
def help():
    return render_template("help.html")

@app.route("/dashboard")
@login_required
def dashboard():

    page = request.args.get("page", 1, type=int)
    filter_type = request.args.get("filter", "all")
    per_page = 5
    all_documents = list(
        predictions_collection.find({
            "user_id": current_user.id,
            "type": {"$ne": "note"}
        }).sort("_id", -1)
    )
    all_documents = clean_mongo_data(all_documents)
    if filter_type != "all":
        now = datetime.now()
        if filter_type == "7":
            limit_date = now - timedelta(days=7)
        elif filter_type == "30":
            limit_date = now - timedelta(days=30)
        filtered_docs = []
        for doc in all_documents:
            try:
                doc_time = datetime.strptime(doc["timestamp"], "%Y-%m-%d %H:%M")
                if doc_time >= limit_date:
                    filtered_docs.append(doc)
            except:
                pass
        all_documents = filtered_docs
    category_counts = {}
    for doc in all_documents:
        cat = doc.get("main_category", "Unknown")
        if cat not in category_counts:
            category_counts[cat] = 0

        category_counts[cat] += 1

    act_counts = {}
    for doc in all_documents:
        for act in doc.get("acts", []):
            act_counts[act] = act_counts.get(act, 0) + 1
    most_common_act = None
    if act_counts:
        most_common_act = max(act_counts, key=act_counts.get)

    most_common_type = None
    if category_counts:
        most_common_type = max(category_counts, key=category_counts.get)
    top_domain = most_common_type
    total_cases = sum(category_counts.values())
    category_percent = {}
    for k, v in category_counts.items():
        if total_cases == 0:
            category_percent[k] = 0
        else:
            category_percent[k] = round((v / total_cases) * 100, 1)

    confidences = [
        doc.get("confidence", 0)
        for doc in all_documents
        if doc.get("confidence")
    ]
    if confidences:
        avg_confidence = round(sum(confidences) / len(confidences))
    else:
        avg_confidence = 0

    total_docs = len(all_documents)
    start = (page - 1) * per_page
    end = start + per_page
    documents = all_documents[start:end]
    total_pages = (total_docs + per_page - 1) // per_page
    chart_labels = list(category_counts.keys())
    chart_data = list(category_counts.values())

    return render_template(
        "dashboard.html",
        documents=documents,
        total=total_cases,
        category_counts=category_counts,
        category_percent=category_percent,
        chart_labels=chart_labels,
        chart_data=chart_data,
        most_common_type=most_common_type,
        total_pages=total_pages,
        current_page=page,
        top_domain=top_domain,
        avg_confidence=avg_confidence,
        most_common_act=most_common_act,
        filter_type=filter_type
    )

@app.route("/view/<id>")
@login_required
def view_prediction(id):
    document = predictions_collection.find_one({"_id": ObjectId(id)})
    return render_template("view_document.html", doc=document)

@app.route("/analyze/<id>")
@login_required
def analyze_document(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id)})
    if not doc:
        return redirect(url_for("dashboard"))
    result = analyze_subdivisions_llm(
        doc["text"],
        doc["main_category"]
    )
    subdivisions = result.get("subdivisions", [])
    return render_template(
        "analysis_result.html", 
        main_category=doc["main_category"], 
        subdivisions=subdivisions 
    )

@app.route("/delete/<id>", methods=["POST"])
@login_required
def delete_prediction(id):
    predictions_collection.delete_one({"_id": ObjectId(id)})
    flash("Case deleted successfully!", "success")

    return redirect(url_for("dashboard"))

@app.route("/export/<id>")
@login_required
def export_pdf(id):
    doc = predictions_collection.find_one({"_id": ObjectId(id)})
    if not doc:
        return redirect(url_for("dashboard"))
    file_path = f"report_{id}.pdf"
    pdf = SimpleDocTemplate(file_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Case Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Main Category: {doc.get('main_category')}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {doc.get('confidence')}%", styles["Normal"]))
    elements.append(Paragraph(f"Status: {doc.get('status')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Detected Subdivisions:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 6))

    subdivisions = doc.get("subdivisions", [])
    if subdivisions:
        sub_list = []
        for sub in subdivisions:
            sub_text = f"{sub.get('title','Issue')} ({sub.get('law','N/A')}) - Severity: {sub.get('severity','Medium')}"
            sub_list.append(ListItem(Paragraph(sub_text, styles["Normal"])))
        elements.append(ListFlowable(sub_list))
    else:
        elements.append(Paragraph("No subdivisions detected.", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Original Text:</b>", styles["Heading3"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(doc.get("text")[:3000], styles["Normal"]))
        pdf.build(elements)
        return send_file(file_path, as_attachment=True)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        new_name = request.form.get("name")
        new_dob = request.form.get("dob")
        new_gender = request.form.get("gender")
        new_email = request.form.get("email")
        new_mobile = request.form.get("mobile")
        print("DOB RECEIVED:", new_dob)
        existing_user = users_collection.find_one({
            "email": new_email,
            "_id": {"$ne": ObjectId(current_user.id)}
        })
        if existing_user:
            flash("Email already in use!", "danger")
            return redirect(url_for("profile"))
        users_collection.update_one(
            {"_id": ObjectId(current_user.id)},
            {
                "$set": {
                    "name": new_name,
                    "dob": new_dob,
                    "gender": new_gender,
                    "email": new_email,
                    "mobile": new_mobile
                }
            }
        )
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    total_predictions = predictions_collection.count_documents(
        {"user_id": current_user.id}
    )
    return render_template(
        "profile.html",
        total_predictions=total_predictions
    )

@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    old_password = request.form.get("old_password")
    new_password = request.form.get("new_password")
    user = users_collection.find_one({"_id": ObjectId(current_user.id)})
    if not check_password_hash(user["password"], old_password):
        flash("Old password incorrect!", "danger")
        return redirect(url_for("profile"))
    hashed_password = generate_password_hash(new_password)
    users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"password": hashed_password}}
    )
    flash("Password updated successfully!", "success")
    return redirect(url_for("profile"))

@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    users_collection.delete_one({"_id": ObjectId(current_user.id)})
    logout_user()
    return redirect(url_for("auth.register"))

@app.route("/update_status/<id>/<new_status>", methods=["POST"])
@login_required
def update_status(id, new_status):
    if new_status not in ["Approved", "Rejected"]:
        return redirect(url_for("dashboard"))
    predictions_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"status": new_status}}
    )
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)