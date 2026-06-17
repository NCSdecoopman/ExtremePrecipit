import fitz

def inspect_drawings(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    drawings = page.get_drawings()
    print(f"Total drawings in {pdf_path}: {len(drawings)}")
    # Print drawings near the top (y < 50)
    for i, draw in enumerate(drawings):
        rect = draw["rect"]
        if rect.y1 < 50:
            print(f"Drawing {i}: type={draw['type']}, rect={rect}, color={draw.get('color')}, fill={draw.get('fill')}")

if __name__ == "__main__":
    inspect_drawings("perso/v2_modifs/fig8.pdf")
