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

import sklearn.feature_selection

# A function for automatically selecting the attributes to keep in a data set.
#
# returns the transformed training and testing set attribute values
# containing only the selected attributes (with all other attributes removed)
def feature_selection(training_X, training_y, testing_X, direction="forward"):
    # create the feature selection algorithm
    feature_selector = sklearn.feature_selection.SequentialFeatureSelector(sklearn.svm.SVR(kernel="rbf"), direction=direction)

    # determine the best attributes to keep using only the training set 
    # (since the testing set wouldn't be available during training)
    new_training_X = feature_selector.fit_transform(training_X, training_y)

    # keep only the selected attributes in the testing set
    new_testing_X = feature_selector.transform(testing_X)

    # determine the attributes chosen by the algorithm:
    # this is a list of True/False values, one per original attribute, where True indicates the original attribute was selected
    chosen = feature_selector.support_

    # to print out only the names of the selected attributes
    print("Columns Selected:")
    for i in range(len(chosen)):
        if chosen[i]: print(training_X.columns[i])

    # return the transformed training and testing set instances
    return new_training_X, new_testing_X

def split_data(data_set, train_percentage, seed):
    # your code goes here
    np.random.seed(seed)
    train = data_set.sample(frac=train_percentage, random_state=seed)
    test = data_set.drop(train.index)
    train_X, train_Y = train.drop("label", axis=1), train["label"]
    test_X, test_Y = test.drop("label", axis=1), test["label"]
    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    dataset = pd.read_csv("data/in/combined_data_with_hashes_and_averages.csv")
    training_X, training_y, testing_X, testing_y = split_data(dataset, 0.75, 1234)
    training_X, testing_X = feature_selection(training_X, training_y, testing_X)
# combine_and_average()
#combine_posts()
# use_both()