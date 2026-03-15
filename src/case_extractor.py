import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extract full text from a judgment PDF
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                
    return full_text