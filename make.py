import subprocess
import sys

scripts = [
    "preprocessing.py",
    "retrieval.py",
    "regressor.py",
    "explanation.py"
]

for script in scripts:
    print(f"Running {script}...")
    
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error: {script} exited with code {result.returncode}")
        break

print("Done.")