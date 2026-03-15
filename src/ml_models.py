import re
import spacy

def extract_case_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    appellant = None
    respondent = None
    for i, line in enumerate(lines):
        if "appellant" in line.lower() and i > 0:
            appellant = lines[i-1]

        if "respondent" in line.lower() and i > 0:
            respondent = lines[i-1]

        if appellant and respondent:
            case = f"{appellant.title()} vs {respondent.title()}"
            case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
            return case.strip()

    for i, line in enumerate(lines):
        if line.lower() == "versus" and i > 0 and i < len(lines)-1:
            party1 = lines[i-1]
            party2 = lines[i+1]
            case = f"{party1.title()} vs {party2.title()}"
            case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
            return case.strip()

    match = re.search(r'(.+?)\s+vs\.?\s+(.+)', text, re.IGNORECASE)
    if match:
        case = f"{match.group(1).title()} vs {match.group(2).title()}"
        case = re.sub(r"\(.*?slp.*?\)", "", case, flags=re.I)
        return case.strip()
    return "Unknown Case"

def extract_court_name(text):
    courts = [
        "SUPREME COURT OF INDIA",
        "DELHI HIGH COURT",
        "BOMBAY HIGH COURT",
        "MADRAS HIGH COURT",
        "KARNATAKA HIGH COURT",
        "CALCUTTA HIGH COURT",
        "HIGH COURT",
        "DISTRICT COURT",
        "SESSIONS COURT",
        "FAMILY COURT",
        "NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION",
        "TRIBUNAL"
    ]
    for court in courts:
        if court.lower() in text.lower():
            return court.title()
    return "Court Not Identified"

def extract_judge_name(text):

    patterns = [
        r"hon'?ble\s+mr\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"hon'?ble\s+ms\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"hon'?ble\s+dr\.?\s+justice\s+([A-Za-z\.\s]+)",
        r"justice\s+([A-Za-z\.\s]+)",
        r"coram\s*[:\-]?\s*([A-Za-z\.\s,]+)",
        r"before\s*[:\-]?\s*([A-Za-z\.\s,]+)"
    ]
    judges = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            name = m.strip()
            name = re.sub(r"\s+", " ", name)
            if 3 < len(name) < 60:
                judges.append(name.title())

    judges = list(set(judges))
    if judges:
        return ", ".join(judges[:2])

    return "Judge Not Identified"

def extract_legal_sections(text):
    pattern = r"(Section\s+\d+[A-Za-z]*\s*(IPC|CrPC|BNSS|CPC|POCSO|IT Act)?|Article\s+\d+|Rule\s+\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    sections = []
    for m in matches:
        if isinstance(m, tuple):
            sections.append(m[0])
        else:
            sections.append(m)
    sections = list(set(sections))
    return sorted(sections)

def extract_parties(text):
    """
    Extract petitioner and respondent names from case title
    """
    match = re.search(r"(.+?)\s+vs\.?\s+(.+)", text, re.IGNORECASE)
    if match:
        petitioner = match.group(1).strip().title()
        respondent = match.group(2).strip().title()
        return petitioner, respondent
    return None, None

def extract_dates(text):
    """
    Extract important dates from legal document
    """
    patterns = [
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}",   
        r"\d{1,2}/\d{1,2}/\d{4}",         
        r"\d{4}-\d{2}-\d{2}"              
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return list(set(dates))[:5]

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from legal document
    """
    doc = nlp(text)
    entities = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": []
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["persons"].append(ent.text)

        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)

        elif ent.label_ == "GPE":
            entities["locations"].append(ent.text)

        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)

    for key in entities:
        entities[key] = list(set(entities[key]))[:5]

    return entities