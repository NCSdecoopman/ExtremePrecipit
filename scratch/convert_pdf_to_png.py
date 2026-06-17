import fitz # PyMuPDF
import os

def convert_pdf(pdf_path, out_png, out_txt):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    pix.save(out_png)
    print(f"Saved PNG to {out_png}")
    
    with open(out_txt, "w", encoding="utf-8") as f:
        text_instances = page.get_text("blocks")
        for inst in text_instances:
            f.write(f"rect: {inst[:4]} | text: {inst[4].strip()}\n")
    print(f"Saved text layout to {out_txt}")

if __name__ == "__main__":
    out_dir = r"C:\Users\nicod\.gemini\antigravity-ide\brain\f9fd9ddd-0a95-423b-9e0a-cfbb4edb6952"
    convert_pdf(r"perso/v2_modifs/fig8.pdf", os.path.join(out_dir, "fig8_preview.png"), "scratch/fig8_layout.txt")
    convert_pdf(r"perso/v2_modifs/fig9.pdf", os.path.join(out_dir, "fig9_preview.png"), "scratch/fig9_layout.txt")
