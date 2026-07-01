import subprocess
import shutil
import os
import sys
from pathlib import Path

# Files map: (qmd_filename, target_filename_in_v2_modifs)
figures_map = [
    ("fig3_nr.qmd", "fig3.pdf"),
    ("fig5.qmd", "fig5.pdf"),
    ("fig6.qmd", "fig6.pdf"),
    ("annexe_1_nr10_quot.qmd", "figa1.pdf"),
    ("annexe_2_nr10_horair.qmd", "figa2.pdf"),
    ("fig7.qmd", "figc1.pdf")
]

print("Starting compilation of all figures...")

# 1. Run assemble_fig3.py manually
print("\n[1/7] Assembling subfigures for fig3...")
res_fig3 = subprocess.run([".venv/Scripts/python.exe", "assemble_fig3.py"], cwd="article", capture_output=True, text=True)
if res_fig3.returncode != 0:
    print(f"Error running assemble_fig3.py: {res_fig3.stderr}")
    sys.exit(1)
print("Success.")

# 2. Run compilation for each figure using compile_qmd.py
for i, (qmd, target) in enumerate(figures_map):
    print(f"\n[{i+2}/7] Compiling {qmd}...")
    res = subprocess.run([".venv/Scripts/python.exe", "compile_qmd.py", qmd], cwd="article", capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling {qmd}: {res.stderr}")
        print(f"Stdout: {res.stdout}")
        sys.exit(1)
    print(f"Successfully compiled {qmd}.")

# 3. Copy compiled PDFs to target folder
print("\nCopying compiled PDFs to perso/v2_modifs/...")
dest_dir = Path("perso/v2_modifs")
dest_dir.mkdir(parents=True, exist_ok=True)

for qmd, target in figures_map:
    pdf_source_name = qmd.replace(".qmd", ".pdf")
    src_path = Path("article") / pdf_source_name
    dest_path = dest_dir / target
    
    if src_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        shutil.copy(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")
    else:
        print(f"Error: {src_path} not found!")
        sys.exit(1)

print("\nAll figures compiled and copied successfully!")
