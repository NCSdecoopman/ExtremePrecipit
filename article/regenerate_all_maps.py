import subprocess
import sys

commands = [
    # 1. stats numday
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs", "--data_type", "stats", "--col_calculate", "numday", "--echelle", "quotidien", "--season", "hydro", "--sat", "99.9"],
    
    # 2. stats mean
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs", "--data_type", "stats", "--col_calculate", "mean", "--echelle", "quotidien", "--season", "hydro", "--sat", "99.0"],
    
    # 3. gev zTpa quotidien
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs_nr", "--data_type", "gev", "--col_calculate", "zTpa", "--echelle", "quotidien", "--season", "hydro", "ond", "jfm", "amj", "jas", "--sat", "99.0"],
    
    # 4. gev zTpa horaire
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs_nr", "--data_type", "gev", "--col_calculate", "zTpa", "--echelle", "horaire", "--season", "hydro", "ond", "jfm", "amj", "jas", "--sat", "99.0"],
    
    # 5. gev z_T_p quotidien
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs", "--data_type", "gev", "--col_calculate", "z_T_p", "--echelle", "quotidien", "--season", "hydro", "ond", "jfm", "amj", "jas", "--sat", "99.0"],
    
    # 6. gev z_T_p horaire
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs", "--data_type", "gev", "--col_calculate", "z_T_p", "--echelle", "horaire", "--season", "hydro", "ond", "jfm", "amj", "jas", "--sat", "90.0"],
    
    # 7. gev z_T_p monthly hourly
    [".venv/Scripts/python.exe", "-m", "src.pipelines.pipeline_generate_outputs_nr", "--data_type", "gev", "--col_calculate", "z_T_p", "--echelle", "horaire", "--season", "jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec", "--sat", "90.0"]
]

print("Starting regeneration of all maps...")
for i, cmd in enumerate(commands):
    print(f"\n[{i+1}/{len(commands)}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        sys.exit(1)
    else:
        print("Success.")

print("\nAll maps regenerated successfully!")
