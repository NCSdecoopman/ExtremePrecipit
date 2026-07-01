import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

COMMANDS: list[tuple[list[str], str]] = [
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs",
            "--data_type",
            "stats",
            "--col_calculate",
            "numday",
            "--echelle",
            "quotidien",
            "--season",
            "hydro",
            "--sat",
            "99.9",
        ],
        "stats numday (quotidien hydro)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs",
            "--data_type",
            "stats",
            "--col_calculate",
            "mean",
            "--echelle",
            "quotidien",
            "--season",
            "hydro",
            "--sat",
            "99.0",
        ],
        "stats mean (quotidien hydro)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs_nr",
            "--data_type",
            "gev",
            "--col_calculate",
            "zTpa",
            "--echelle",
            "quotidien",
            "--season",
            "hydro",
            "ond",
            "jfm",
            "amj",
            "jas",
            "--sat",
            "99.0",
        ],
        "gev zTpa quotidien (saisons)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs_nr",
            "--data_type",
            "gev",
            "--col_calculate",
            "zTpa",
            "--echelle",
            "horaire",
            "--season",
            "hydro",
            "ond",
            "jfm",
            "amj",
            "jas",
            "--sat",
            "99.0",
        ],
        "gev zTpa horaire (saisons)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs",
            "--data_type",
            "gev",
            "--col_calculate",
            "z_T_p",
            "--echelle",
            "quotidien",
            "--season",
            "hydro",
            "ond",
            "jfm",
            "amj",
            "jas",
            "--sat",
            "99.0",
        ],
        "gev z_T_p quotidien (saisons)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs",
            "--data_type",
            "gev",
            "--col_calculate",
            "z_T_p",
            "--echelle",
            "horaire",
            "--season",
            "hydro",
            "ond",
            "jfm",
            "amj",
            "jas",
            "--sat",
            "90.0",
        ],
        "gev z_T_p horaire (saisons)",
    ),
    (
        [
            "-m",
            "src.pipelines.pipeline_generate_outputs_nr",
            "--data_type",
            "gev",
            "--col_calculate",
            "z_T_p",
            "--echelle",
            "horaire",
            "--season",
            "jan",
            "fev",
            "mar",
            "avr",
            "mai",
            "jui",
            "juill",
            "aou",
            "sep",
            "oct",
            "nov",
            "dec",
            "--sat",
            "90.0",
        ],
        "gev z_T_p horaire (mensuel)",
    ),
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


def run_command(payload: tuple[int, list[str], str, dict[str, str]]) -> tuple[int, str, int, str, str]:
    index, module_args, label, env = payload
    cmd = [PYTHON, "-u", *module_args]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return index, label, result.returncode, result.stdout, result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all map outputs in parallel.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(len(COMMANDS), os.cpu_count() or 1),
        help="Number of pipeline jobs to run in parallel (default: min(7, CPU count)).",
    )
    args = parser.parse_args()
    max_workers = max(1, min(args.jobs, len(COMMANDS)))
    env = thread_env(max_workers)

    print(
        f"Starting regeneration of {len(COMMANDS)} map jobs "
        f"with {max_workers} parallel workers on {os.cpu_count()} logical CPUs..."
    )
    print(f"BLAS/OpenMP threads per job: {env['OMP_NUM_THREADS']}")

    payloads = [
        (index, module_args, label, env)
        for index, (module_args, label) in enumerate(COMMANDS, start=1)
    ]

    failures: list[tuple[int, str, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_command, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            index, label, returncode, stdout, stderr = future.result()
            if returncode == 0:
                print(f"[{index}/{len(COMMANDS)}] OK  {label}")
            else:
                print(f"[{index}/{len(COMMANDS)}] ERR {label}")
                if stdout.strip():
                    print(stdout[-4000:])
                if stderr.strip():
                    print(stderr[-4000:], file=sys.stderr)
                failures.append((index, label, stderr or stdout))

    if failures:
        print("\nFailed jobs:")
        for index, label, message in sorted(failures):
            print(f"  [{index}] {label}")
        sys.exit(1)

    print("\nAll maps regenerated successfully!")


if __name__ == "__main__":
    main()
