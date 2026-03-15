from src.case_extractor import extract_text_from_pdf
from src.rag_engine import retrieve_relevant_laws
import re
import pandas as pd

ipc_data = pd.read_csv("data/ipc_sections.csv")

def extract_case_name(text):
    """
    Try to detect case name from judgment text
    """
    lines = text.split("\n")
    for line in lines[:50]:
        line_lower = line.lower()
        if " vs " in line_lower or " v. " in line_lower or " vs. " in line_lower:
            return line.strip()
        if "petitioner" in line_lower and "respondent" in line_lower:
            return line.strip()
    return "Case Name Not Found"

def analyze_case(pdf_path):
    case_text = extract_text_from_pdf(pdf_path)
    case_name = extract_case_name(case_text)
    results = retrieve_relevant_laws(case_text, k=5)
    ipc_sections = detect_ipc_sections(case_text)
    issues = extract_legal_issues(case_text)
    domain = detect_domain(case_text)
    sections = detect_sections(case_text)
    citations = extract_case_citations(case_text)
    if case_name == "Case Name Not Found" and len(results) > 0:
        case_name = results[0]["title"]
        detected_acts = []
        legal_provisions = []
        similar_cases = []
        for r in results:
            text = r["content"].lower()
            if "ipc" in text:
                detected_acts.append("Indian Penal Code")
            if "civil procedure" in text:
                detected_acts.append("Code of Civil Procedure")
                if "section" in text:
                    legal_provisions.append(r["title"])
                    similar_cases.append(r["title"])
                    output = {
                        "domain": domain,
                        "case_name": case_name,
                        "citations": citations,
                        "detected_acts": ["Indian Penal Code"] if ipc_sections else list(set(detected_acts)),
                        "legal_provisions": sections if sections else (ipc_sections if ipc_sections else legal_provisions[:3]),
                        "legal_issues": issues if issues else [
                            "Legal dispute between parties",
                            "Application of law",
                            "Judicial interpretation required"
                        ],
                        "similar_judgements": similar_cases[:3]
                    }
                    return output

def detect_ipc_sections(text):
    detected_sections = []
    text_lower = text.lower()
    ipc_map = {
        str(row["Offense"]).lower(): str(row["Section"])
        for _, row in ipc_data.iterrows()
    }
    for offense, section in ipc_map.items():
        if offense in text_lower:
            detected_sections.append(section)

    return list(set(detected_sections))[:3]

def extract_case_citations(text):
    """
    Extract common Indian case citation formats
    """
    patterns = [
        r"\(\d{4}\)\s*\d+\s*SCC\s*\d+",       
        r"AIR\s*\d{4}\s*SC\s*\d+",            
        r"\d{4}\s*SCC\s*Online\s*\d+",        
        r"\(\d{4}\)\s*\d+\s*SCR\s*\d+"        
    ]
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return list(set(citations))[:5]

def extract_legal_issues(text):
    issues = []
    text_lower = text.lower()
    if "property" in text_lower:
        issues.append("Property ownership dispute")

    if "sale deed" in text_lower:
        issues.append("Validity of sale deed")

    if "contract" in text_lower:
        issues.append("Breach of contract")

    if "murder" in text_lower or "kill" in text_lower:
        issues.append("Intent to cause death")

    if "fraud" in text_lower or "cheating" in text_lower:
        issues.append("Financial fraud or cheating")

    if "assault" in text_lower:
        issues.append("Physical assault")

    return issues[:3]

def detect_domain(text):
    text_lower = text.lower()
    if "property" in text_lower or "land" in text_lower:
        return "Property Law"

    if "murder" in text_lower or "assault" in text_lower or "crime" in text_lower:
        return "Criminal Law"

    if "internet" in text_lower or "hacking" in text_lower or "cyber" in text_lower:
        return "Cyber Law"

    if "marriage" in text_lower or "divorce" in text_lower:
        return "Family Law"

    if "contract" in text_lower or "agreement" in text_lower:
        return "Contract Law"

    return "General Law"

def detect_sections(text):
    sections = re.findall(r"section\s+\d+", text.lower())
    formatted_sections = []
    for s in sections:
        num = s.split()[1]
        formatted_sections.append(f"Section {num}")

    return list(set(formatted_sections))[:3]