import sys
import sqlite3
import subprocess

def main():
    if len(sys.argv) != 6:
        print("Usage: python train_best_hparams.py path/to/database.db path/to/hparams path/to/data path/to/output experiment_name")
        sys.exit(1)

    db_path = sys.argv[1]
    hparams_path = sys.argv[2]
    data_folder = sys.argv[3]
    output_folder = sys.argv[4]
    experiment_name = sys.argv[5]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get top 5 trials based on value
    cursor.execute("""
        SELECT trial_id
        FROM trial_values
        ORDER BY value DESC
        LIMIT 5
    """)
    trial_ids = [row[0] for row in cursor.fetchall()]

    if not trial_ids:
        print("No trials found.")
        return

    # For each trial, get parameters and run scripts
    for trial_id in trial_ids:
        print(f"\n=== Running trial_id {trial_id} ===")

        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))

        params = cursor.fetchall()

        for seed in range(5):
            train_cmd = ["python", "train.py", hparams_path, f"--data_folder={data_folder}", f"--output_folder={output_folder}",
                     "--experiment_name", f"{experiment_name}_trial_{trial_id}", f"--seed={seed}"]
            
            for name, value in params:
                train_cmd.extend([f"--{name}", str(value)])

            print("Running:", " ".join(train_cmd))
            subprocess.run(train_cmd, check=True)

            print("Running permutation_test.py")
            subprocess.run(["python", "permutation_test.py", f"{output_folder}/{experiment_name}_trial_{trial_id}/seed_{seed}", f"{output_folder}/{experiment_name}_trial_{trial_id}/seed_{seed}"], check=True)

    conn.close()
    print("\nAll trials completed successfully.")

if __name__ == "__main__":
    main()
