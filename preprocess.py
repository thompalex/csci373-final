import pandas as pd
import numpy as np

def combine_and_average():
    automated = pd.read_json("data/in/automatedAccountData.json")
    nonautomated = pd.read_json("data/in/nonautomatedAccountData.json")
    dataset = pd.concat([automated, nonautomated])
    processed_dataset = pd.DataFrame()
    for column in dataset.columns:
        if column == "automatedBehaviour": continue
        print(column)
        if dataset[column].dtype == 'object':
            processed_dataset[f"avg_{column}"] = [np.mean(x) if len(x) > 0 else np.nan for x in dataset[column]]
            processed_dataset[f"std_{column}"] = [np.std(x) if len(x) > 0 else np.nan for x in dataset[column]]
            
        else:
            processed_dataset[column] = dataset[column]
    processed_dataset["label"] = dataset["automatedBehaviour"]
    processed_dataset.to_csv("data/in/combined_data_with_averages.csv", index=False)
    return processed_dataset

def combine_posts(dataset):
    media_columns = [column for column in dataset.columns if column.startswith("media")]
    dataset.fillna(-1, inplace=True)
    combined_posts_list = []
    for ind, user in dataset.iterrows():
        # print(user)
        combined_posts = zip(*[user[column] for column in media_columns])
        combined_posts = [hash(tuple(post)) for post in combined_posts]
        # dataset.at[ind, 'combined_posts'] = hash(tuple(combined_posts))
        combined_posts_list.append(hash(tuple(combined_posts)))
    dataset.drop(media_columns, axis=1, inplace=True)
    dataset['combined_posts'] = combined_posts_list
    # dataset.drop("automatedBehaviour", axis=1, inplace=True)
    dataset.to_csv("data/in/combined_data_with_hashes.csv", index=False)
    return combined_posts_list

def use_both():
    processed_dataset = combine_and_average()
    processed_dataset["combined_posts"] = combine_posts(processed_dataset)
    processed_dataset.to_csv("data/in/combined_data_with_hashes_and_averages.csv", index=False)

# combine_and_average()
#combine_posts()
use_both()