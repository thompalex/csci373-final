import pandas as pd
import numpy as np

def combine_and_average():
    automated = pd.read_json("data/in/automatedAccountData.json")
    nonautomated = pd.read_json("data/in/nonautomatedAccountData.json")
    dataset = pd.concat([automated, nonautomated])
    processed_dataset = pd.DataFrame()
    for column in dataset.columns:
        if column.lower() == "automatedBehaviour": continue
        print(column)
        if dataset[column].dtype == 'object':
            processed_dataset[f"avg_{column}"] = [np.mean(x) for x in dataset[column]]
            processed_dataset[f"std_{column}"] = [np.std(x) for x in dataset[column]]
        else:
            processed_dataset[column] = dataset[column]
    processed_dataset["label"] = dataset["automatedBehaviour"]
    processed_dataset.to_csv("data/out/combined_data_with_averages.csv", index=False)

combine_and_average()