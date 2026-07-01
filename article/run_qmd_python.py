import sys
import re
import types
from pathlib import Path

# Create a mock cairosvg module using svglib and reportlab to avoid Cairo DLL dependency issues
cairosvg_mock = types.ModuleType("cairosvg")

def svg2pdf(url=None, write_to=None, **kwargs):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    import re
    from pathlib import Path
    
    svg_path = Path(url or kwargs.get('src') or kwargs.get('url'))
    pdf_path = Path(write_to)
    
    content = svg_path.read_text(encoding="utf-8")
    cleaned = re.sub(r'<!DOCTYPE[^>]*>', '', content)
    temp_svg = svg_path.parent / (svg_path.name + "_temp_cleaned.svg")
    temp_svg.write_text(cleaned, encoding="utf-8")
    try:
        drawing = svg2rlg(str(temp_svg))
        renderPDF.drawToFile(drawing, str(pdf_path))
    finally:
        if temp_svg.exists():
            temp_svg.unlink()

cairosvg_mock.svg2pdf = svg2pdf
sys.modules["cairosvg"] = cairosvg_mock


def run_qmd_python(qmd_path: Path):
    print(f"Extracting and running Python from {qmd_path}...")
    content = qmd_path.read_text(encoding="utf-8")
    
    # Find all ```{python} ... ``` blocks
    pattern = re.compile(r'```\{python\}(.*?)\n```', re.DOTALL)
    blocks = pattern.findall(content)
    
    for i, block in enumerate(blocks):
        print(f"Running Python block {i+1}...")
        # Remove chunk options like #| include: false
        lines = block.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip().startswith('#|'):
                continue
            clean_lines.append(line)
        
        code = '\n'.join(clean_lines)
        
        # Execute the code in the local context
        # We define a shared globals dictionary to keep state between blocks
        exec(code, globals())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_qmd_python.py <path_to_qmd>")
        sys.exit(1)
    run_qmd_python(Path(sys.argv[1]))
