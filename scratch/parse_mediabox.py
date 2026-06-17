import sys

def parse_mediabox(pdf_path):
    print(f"--- MediaBox for {pdf_path} ---")
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        # Look for /MediaBox [x y w h]
        matches = list(set(re.findall(b'/MediaBox\s*\[\s*([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s*\]', content)))
        if not matches:
            # Try without backslash or different spacing
            matches = list(set(re.findall(b'/MediaBox\[([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\]', content)))
            
        for m in matches:
            w = float(m[2]) - float(m[0])
            h = float(m[3]) - float(m[1])
            print(f"  MediaBox: { [float(x) for x in m] } (width={w:.1f}, height={h:.1f})")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    import re
    parse_mediabox(r"perso/v2_modifs/fig9.pdf")
    parse_mediabox(r"perso/v2_modifs/fig4.pdf")
    parse_mediabox(r"scratch/fig8_test.pdf")

