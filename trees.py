import sys
import time
from preprocess import feature_selection, split_data
import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def scale_dataset(dataset):
    scaled_dataset = dataset.copy()
    for column in dataset.columns:
        if column.lower() == "label" or dataset[column].dtype=='object': continue
        minval, maxval = min(scaled_dataset[column]), max(scaled_dataset[column])
        if minval != maxval:
            scaled_dataset[column] = (scaled_dataset[column] - minval) / (maxval - minval)
        else:
            scaled_dataset[column] = 0
    return scaled_dataset

def create_decision_tree_get_accuracy(X_train, X_test, y_train, y_test):
    # Training the Decision Tree model on the Training set
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    a = accuracy_score(y_test, y_pred)
    return a


def create_random_forest_get_accuracy(X_train, X_test, y_train, y_test, estimate):
    # Training the Decision Tree model on the Training set
    classifier = RandomForestClassifier(n_estimators=estimate)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    a = accuracy_score(y_test, y_pred)
    return a

def find_best_trees(output_name, X_train, X_test, y_train, y_test):
    # Define the list of tree amounts
    tree_amounts = [5, 10, 25, 50, 100]
    # Initialize lists to store the accuracies and number of trees
    accuracies = []

    # Call the random forest accuracy function for each tree amount
    for amount in tree_amounts:
        accuracy = create_random_forest_get_accuracy(X_train, X_test, y_train, y_test, amount)
        accuracies.append(accuracy)
        print(f"Random Forest Accuracy for {amount} trees: {accuracy}")
    # Find the best accuracy and corresponding number of trees
    best_accuracy = max(accuracies)
    best_num_trees = tree_amounts[accuracies.index(best_accuracy)]
    num_trees = ['5', '10', '25', '50', '100']
    # Display the results in a bar chart
    plt.bar(num_trees, accuracies)
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy for Different Number of Trees')
    plt.savefig(output_name)
    plt.show()

    # Save the best accuracy and number of trees
    best_accuracy_and_num_trees = {'Best Accuracy': best_accuracy, 'Number of Trees': best_num_trees}
    print("Best Accuracy and Number of Trees: ", best_accuracy_and_num_trees)
    return best_accuracy



def without_feature(X_train, X_test, y_train, y_test):
    # Get decision tree accuracy
    decision_tree_accuracy = create_decision_tree_get_accuracy(X_train, X_test, y_train, y_test)
    print("Decision Tree Accuracy: ", decision_tree_accuracy)
    best_acc = find_best_trees("Best_trees_without_.png", X_train, X_test, y_train, y_test)
    compare_accuracy(decision_tree_accuracy, best_acc, "Tree_Forest_Comparison.png")


def with_feature(X_train, X_test, y_train, y_test):
    X_train, X_test = feature_selection(X_train, y_train, X_test, "backward")
    # Get decision tree accuracy
    decision_tree_accuracy = create_decision_tree_get_accuracy(X_train, X_test, y_train, y_test)
    print("Decision Tree Accuracy: ", decision_tree_accuracy)
    best_acc = find_best_trees("Best_trees_with_.png", X_train, X_test, y_train, y_test)
    compare_accuracy(decision_tree_accuracy, best_acc, "Tree_Forest_Comparison_wFeature.png")


def compare_accuracy(decision_tree_accuracy, random_forest_accuracy, output_name):
    # Define the labels and heights for the bar chart
    labels = ['Decision Tree', 'Random Forest']
    heights = [decision_tree_accuracy, random_forest_accuracy]

    # Create the bar chart
    plt.bar(labels, heights)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Decision Tree and Random Forest Accuracy')
    plt.savefig(output_name)
    plt.show()



def main():
    random_seed = 1234
    training_precentage = 0.75
    dataset_name = "data/in/combined_data_with_hashes_and_averages.csv"
    print("Dataset: ", dataset_name)
    # Importing the dataset
    dataset = pd.read_csv(dataset_name)
    # dataset = convert_labels(dataset)
    dataset = scale_dataset(dataset)

    # Splitting the dataset into the Training set and Test set
    X_train, y_train, X_test, y_test = split_data(dataset, training_precentage, random_seed)

    without_feature(X_train, X_test, y_train, y_test)
    with_feature(X_train, X_test, y_train, y_test)

# random_forest_accuracy = create_random_forest_get_accuracy(X_train, X_test, y_train, y_test)
# print("Random Forest Accuracy: ", random_forest_accuracy)   
# print("- Finished Random Forest")

main()
