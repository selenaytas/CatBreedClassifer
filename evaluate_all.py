import os
from evaluate import evaluate_experiment

def run_all_evaluations():
    exp_root = "experiments"
    experiments = sorted(os.listdir(exp_root))

    print("\n🚀 Tüm deneyler evaluate ediliyor...")

    for exp in experiments:
        folder = os.path.join(exp_root, exp)
        if not os.path.isdir(folder):
            continue

        print(f"\n==============================")
        print(f"📊 Evaluate: {exp}")
        print(f"==============================")

        evaluate_experiment(exp)

if __name__ == "__main__":
    run_all_evaluations()
