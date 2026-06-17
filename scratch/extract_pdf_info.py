import sys
import re

def extract_pdf_fonts_and_meta(pdf_path):
    print(f"--- Parsing PDF {pdf_path} ---")
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
            
        # Try to find /BaseFont
        base_fonts = re.findall(b'/BaseFont\s*/([A-Za-z0-9+-]+)', content)
        if base_fonts:
            print("Found BaseFonts:")
            unique_fonts = sorted(list(set(base_fonts)))
            for f in unique_fonts:
                print(f"  {f.decode('utf-8', errors='ignore')}")
        else:
            print("No BaseFonts found.")
            
        # Try to find Creator or Producer
        creator = re.search(b'/Creator\s*\((.*?)\)', content)
        if creator:
            print(f"Creator: {creator.group(1).decode('utf-8', errors='ignore')}")
        producer = re.search(b'/Producer\s*\((.*?)\)', content)
        if producer:
            print(f"Producer: {producer.group(1).decode('utf-8', errors='ignore')}")
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"perso/v2_modifs/fig9.pdf"
    extract_pdf_fonts_and_meta(pdf_path)
