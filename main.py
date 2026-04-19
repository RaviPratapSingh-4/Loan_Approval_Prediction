import subprocess
import sys


def run(script):
    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print('='*50)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\nFailed at: {script}")
        sys.exit(1)
    print(f"Done: {script}")


if __name__ == "__main__":
    run("pipelines/data_pipeline.py")
    run("pipelines/feature_engineering.py")
    run("pipelines/feature_selector.py")
    run("pipelines/split_and_transform.py")
    run("training/train.py")
    run("evaluation/fairness_analysis.py")
    run("evaluation/robustness.py")
    print("\nFull pipeline complete.")