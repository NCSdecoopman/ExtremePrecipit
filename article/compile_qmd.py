import sys
import re
import os
import subprocess
from pathlib import Path

def get_clean_qmd_content(qmd_path: Path) -> str:
    content = qmd_path.read_text(encoding="utf-8")
    
    # Replace all ```{python} ... ``` blocks with empty string
    pattern = re.compile(r'```\{python\}(.*?)\n```', re.DOTALL)
    clean_content = pattern.sub('', content)
    return clean_content

def compile_qmd(qmd_name: str):
    qmd_path = Path(qmd_name)
    if not qmd_path.exists():
        print(f"Error: {qmd_path} does not exist.")
        return False
        
    print(f"\n==========================================")
    print(f"Compiling {qmd_path}...")
    print(f"==========================================")
    
    # 1. Run the Python blocks using the current interpreter
    from run_qmd_python import run_qmd_python
    try:
        run_qmd_python(qmd_path)
    except Exception as e:
        print(f"Error running Python blocks: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 2. Create clean QMD file without Python blocks
    clean_qmd_path = qmd_path.parent / f"temp_{qmd_path.stem}.qmd"
    clean_content = get_clean_qmd_content(qmd_path)
    clean_qmd_path.write_text(clean_content, encoding="utf-8")
    
    # 3. Render clean QMD to PDF using Quarto
    try:
        print(f"Rendering clean QMD to PDF with Quarto...")
        cmd = ["quarto", "render", str(clean_qmd_path), "--to", "pdf"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error rendering PDF: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        if clean_qmd_path.exists():
            clean_qmd_path.unlink()
        return False
        
    # 4. Copy the output PDF to the desired name
    compiled_pdf = qmd_path.parent / f"temp_{qmd_path.stem}.pdf"
    target_pdf = qmd_path.parent / f"{qmd_path.stem}.pdf"
    
    if compiled_pdf.exists():
        if target_pdf.exists():
            target_pdf.unlink()
        compiled_pdf.rename(target_pdf)
        print(f"Success! Created: {target_pdf}")
    else:
        print(f"Error: Compiled PDF not found at {compiled_pdf}")
        
    # 5. Cleanup temp files
    for suffix in [".qmd", ".tex", ".aux", ".log"]:
        temp_file = qmd_path.parent / f"temp_{qmd_path.stem}{suffix}"
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Could not remove temp file {temp_file}: {e}")
                
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile_qmd.py <path_to_qmd>")
        sys.exit(1)
    compile_qmd(sys.argv[1])
