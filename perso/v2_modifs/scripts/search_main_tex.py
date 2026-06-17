import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\nicod\Documents\GitHub\ExtremePrecipit\perso\v1_1_preprint\main.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(header in line for header in ['\\section{', '\\subsection{', '\\subsubsection{']) or '3.4' in line:
        print(f"Line {i+1}: {line.strip()}")



