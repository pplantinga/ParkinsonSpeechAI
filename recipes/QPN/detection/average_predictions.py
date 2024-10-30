import os
import csv
import pandas as pd

# Get experiment name from command line
experiment_name = sys.argv[1]

# Extract the relevant data from the prediction files
data = []
for i in range(1, 6):
    fpath = f'results/{experiment_name}/{i}/predictions.csv'
    with open(fpath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(data) != 0:
                next(reader)
            row["Seed"] = i
            data.append(row)

# Write to a single excel file with means
df = pd.DataFrame(data)

# Get averages and variance
averages = df.mean().tolist()
variances = df.var().tolist()

# Add them to df
df.loc[len(df)] = "Averages" + averages
df.loc[len(df)] = "Variances" + variances

print(df)
df.to_excel(f'{experiment_name}_predictions.xlsx', index=False)
