import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "article"
PYTHON = sys.executable
DEST_DIR = ROOT / "perso" / "v2_modifs" / "final"

FIGURES_MAP: list[tuple[str, str]] = [
    ("fig3_nr.qmd", "fig3.pdf"),
    ("fig5.qmd", "fig5.pdf"),
    ("fig6.qmd", "fig6.pdf"),
    ("fig8.qmd", "fig7.pdf"),
    ("annexe_1_nr10_quot.qmd", "figa1.pdf"),
    ("annexe_2_nr10_horair.qmd", "figa2.pdf"),
    ("fig7.qmd", "figc1.pdf"),
]


def thread_env(max_workers: int) -> dict[str, str]:
    env = os.environ.copy()
    threads = max(1, (os.cpu_count() or 1) // max_workers)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = str(threads)
    return env


def run_subprocess(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def compile_qmd(qmd_name: str, env: dict[str, str]) -> tuple[str, int, str, str]:
    result = run_subprocess([PYTHON, "compile_qmd.py", qmd_name], ARTICLE, env)
    return qmd_name, result.returncode, result.stdout, result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile all article figures in parallel.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(len(FIGURES_MAP), os.cpu_count() or 1),
        help="Number of Quarto compilations to run in parallel.",
    )
    args = parser.parse_args()
    max_workers = max(1, min(args.jobs, len(FIGURES_MAP)))
    env = thread_env(max_workers)

    print(f"Starting compilation of all figures with {max_workers} parallel workers...")

    print("\n[1/3] Assembling subfigures for fig3...")
    res_fig3 = run_subprocess([PYTHON, "assemble_fig3.py"], ARTICLE, env)
    if res_fig3.returncode != 0:
        print(res_fig3.stderr or res_fig3.stdout)
        sys.exit(1)
    print("Success.")

    print("\n[2/3] Generating fig4...")
    res_fig4 = run_subprocess([PYTHON, "generate_fig4.py"], ARTICLE, env)
    if res_fig4.returncode != 0:
        print(res_fig4.stderr or res_fig4.stdout)
        sys.exit(1)
    print("Success.")

    print(f"\n[3/3] Compiling {len(FIGURES_MAP)} QMD files in parallel...")
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compile_qmd, qmd, env): qmd
            for qmd, _ in FIGURES_MAP
        }
        for future in as_completed(futures):
            qmd, returncode, stdout, stderr = future.result()
            if returncode == 0:
                print(f"OK  {qmd}")
            else:
                print(f"ERR {qmd}")
                if stdout.strip():
                    print(stdout[-4000:])
                if stderr.strip():
                    print(stderr[-4000:], file=sys.stderr)
                failures.append((qmd, stderr or stdout))

    if failures:
        sys.exit(1)

    print(f"\nCopying compiled PDFs to {DEST_DIR}...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copy_map = [
        ("fig4.pdf", "fig4.pdf"),
        ("fig3_nr.pdf", "fig3.pdf"),
        ("fig5.pdf", "fig5.pdf"),
        ("fig6.pdf", "fig6.pdf"),
        ("fig8.pdf", "fig7.pdf"),
        ("annexe_1_nr10_quot.pdf", "figa1.pdf"),
        ("annexe_2_nr10_horair.pdf", "figa2.pdf"),
        ("fig7.pdf", "figc1.pdf"),
    ]
    for source_name, target_name in copy_map:
        src_path = ARTICLE / source_name
        dest_path = DEST_DIR / target_name
        if not src_path.exists():
            print(f"Error: {src_path} not found!")
            sys.exit(1)
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_path.name} -> {dest_path}")

    print("\nAll figures compiled and copied successfully!")


if __name__ == "__main__":
    main()
