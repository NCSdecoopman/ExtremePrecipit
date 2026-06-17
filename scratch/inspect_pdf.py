import pypdf
import sys

def inspect_pdf(pdf_path):
    print(f"Reading {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        print(f"--- Page {i+1} ---")
        text = page.extract_text()
        print(text)

if __name__ == "__main__":
    inspect_pdf("perso/v2_modifs/fig9.pdf")
