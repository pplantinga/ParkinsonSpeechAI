import sys
import sqlite3
import subprocess
import json

def decode_param(param_value, distribution_json):
    """
    Decode Optuna param_value using distribution_json.
    """
    dist = json.loads(distribution_json)
    name = dist["name"]
    attrs = dist.get("attributes", {})

    if name == "CategoricalDistribution":
        choices = attrs["choices"]
        return choices[int(float(param_value))]

    elif name == "IntDistribution":
        return int(float(param_value))

    elif name == "FloatDistribution":
        return float(param_value)

    else:
        # Fallback: return raw value
        return param_value


def main():
    if len(sys.argv) != 6:
        print("Usage: python train_best_hparams.py path/to/database.db path/to/hparams path/to/data path/to/storage experiment_name")
        sys.exit(1)

    db_path = sys.argv[1]
    hparams_path = sys.argv[2]
    data_folder = sys.argv[3]
    storage_folder = sys.argv[4]
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

    for trial_id in trial_ids:
        print(f"\n=== Running trial_id {trial_id} ===")

        cursor.execute("""
            SELECT param_name, param_value, distribution_json
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))

        params = cursor.fetchall()

        decoded_params = {}
        for name, value, dist_json in params:
            decoded_params[name] = decode_param(value, dist_json)

        for seed in range(5):
            train_cmd = [
                "python",
                "train.py",
                hparams_path,
                f"--data_folder={data_folder}",
                f"--storage_folder={storage_folder}",
                f"--experiment_name={experiment_name}_trial_{trial_id}",
                f"--seed={seed}",
            ]

            for name, value in decoded_params.items():
                train_cmd.append(f"--{name}={value}")

            print("Running:", " ".join(train_cmd))
            subprocess.run(train_cmd, check=True)

    conn.close()
    print("\nAll trials completed successfully.")


if __name__ == "__main__":
    main()
