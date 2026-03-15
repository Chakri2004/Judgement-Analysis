import os
import fitz
import re

def chunk_text(text, chunk_size=500):
    """
    Split large judgments into smaller semantic chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_sections(text):
    """
    Extract legal section references (IPC / CrPC etc.)
    """
    matches = re.findall(r"section\s+\d+[a-z]*", text.lower())
    return list(set(matches))

def load_cases():
    base_path = "data/supreme_court_judgments"
    cases = []
    for year in os.listdir(base_path):
        year_path = os.path.join(base_path, year)
        if os.path.isdir(year_path):
            for file in os.listdir(year_path):
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(year_path, file)
                    try:
                        doc = fitz.open(file_path)
                        text = ""
                        for page in doc:
                            page_text = page.get_text()
                            if page_text:
                                text += page_text
                        doc.close()
                        if len(text) > 200:
                            chunks = chunk_text(text)
                            for chunk in chunks:
                                cases.append({
                                    "text": chunk,
                                    "sections": extract_sections(chunk)
                                })
                    except Exception as e:
                        print("Skipped:", file)

    return cases