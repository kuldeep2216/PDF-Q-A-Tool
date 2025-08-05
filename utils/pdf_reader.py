from pdfminer.high_level import extract_text

def pdf_to_text(pdf_path):
    try:
        return extract_text(pdf_path)
    except:
        return ""
