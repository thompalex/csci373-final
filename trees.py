import sys
import time
import pandas as pd
import numpy as np
import sklearn.tree, sklearn.ensemble
import matplotlib.pyplot as plt

class DecisionTrees:
    def __init__(self, train_x, train_y, num_trees, labels):
        self.train_x, self.train_y = train_x, train_y
        self.num_trees = num_trees
        self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees) if num_trees > 1 else sklearn.tree.DecisionTreeClassifier()
        self.model.fit(self.train_x, self.train_y)
        self.labels = labels

    def predict(self, test):
        # It will return a prediction for each test instance and maintain the ordering
        preds = self.model.predict(test)
        return preds

    def create_confusion_matrix(self, test, preds):
        # Create the confusion matrix for our predictions
        confusion_matrix = {actual_label: {pred_label: 0 for pred_label in self.labels} for actual_label in self.labels}
        for i in range(len(test)):
            confusion_matrix[test[i]][preds[i]] += 1
        return confusion_matrix

    def print_metrics(self, confusion_matrix):
        # Generate and print the accuracy of our model
        accuracy = sum([confusion_matrix[label][label] for label in self.labels]) / sum([sum(confusion_matrix[label].values()) for label in self.labels])
        recall = {label: confusion_matrix[label][label] / sum([confusion_matrix[label][pred_label] for pred_label in self.labels]) for label in self.labels}
        for label in recall:
            print(f'Recall for {label}: {recall[label]}')
        print(f'Accuracy: {accuracy}')
        return accuracy
        

    def create_output_file(self, confusion_matrix, outfile_name):
        # Create our output file from the confusion matrix
        output = ",".join(self.labels) + ",\n"
        for actual_label in self.labels:
            for pred_label in self.labels:
                output += f'{confusion_matrix[actual_label][pred_label]},'
            output += f'{actual_label}\n'
        outfile_name = outfile_name if outfile_name else f"outfile.csv"
        with open(outfile_name, "w") as file:
            file.write(output)

def log_tree(tree, dataset, dataset_filename, train_percentage, seed):
    # create the filename
    filename = ("tree"
                + "_" + dataset_filename[:-4]
                + "_1t"
                + "_" + str(int(train_percentage * 100)) + "p"
                + "_" + str(seed) + ".png")

    attributes = list(dataset.drop("label", axis=1))
    labels = sorted(list(dataset["label"].unique()))

    fig = plt.figure(figsize=(100, 100))
    plotted = sklearn.tree.plot_tree(tree,
                                     feature_names=attributes,
                                     class_names=labels,
                                     filled=True,
                                     rounded=True)
    fig.savefig(filename)

def load_data(filepath, train_percentage, random_seed):
    np.random.seed(random_seed)  # Set the numpy random seed
    data = pd.read_csv(filepath)
    labels = data["label"].unique()
    train = data.sample(frac=train_percentage, random_state=random_seed)
    test = data.drop(train.index)
    train_X, train_Y = train.drop("label", axis=1), train["label"]
    test_X, test_Y = test.drop("label", axis=1), test["label"]
    return data, train_X, train_Y, test_X, test_Y, labels


def main(filepath, num_trees, train_size, seed, save_tree=False):
    startTime = time.time()
    data, train_x, train_y, test_x, test_y, labels = load_data(filepath, train_size, seed)
    # Create an instance of the DecisionTree class
    model = DecisionTrees(train_x, train_y, num_trees, labels)
    # Make predictions
    preds = model.predict(test_x)
    if save_tree:
        log_tree(model.model, data, filepath, train_size, seed)
    # Evaluate the model
    confusion_matrix = model.create_confusion_matrix(test_y.to_list(), preds)
    accuracy = model.print_metrics(confusion_matrix)
    # Write the output to a file
    outfile_name = f"results_{filepath.split('.')[0]}_{num_trees}t_{int(train_size*100)}p_{seed}.csv"
    model.create_output_file(confusion_matrix, outfile_name)
    print(f"Time taken: {(time.time() - startTime) / 60} minutes")
    return accuracy

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python knn.py <filepath> <num_trees> <train_size> <seed>")
        sys.exit(1)
    # Load in our command line arguments
    filepath = sys.argv[1]
    num_trees = int(sys.argv[2])
    train_size = float(sys.argv[3])
    seed = int(sys.argv[4])
    save_tree = False
    if len(sys.argv) == 6:
        save_tree = bool(sys.argv[5])
    # Run the knn algorithm
    main(filepath, num_trees, train_size, seed, save_tree)
