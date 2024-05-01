import pandas as pd
import sys
import numpy as np
import tensorflow as tf

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


def one_hot_encode(dataset):
    for column in dataset.columns:
        if column.lower() == "label" or dataset[column].dtype!='object': continue
        onehots = pd.get_dummies(dataset[column], column, drop_first=True, dtype=int)
        dataset = pd.concat([dataset.drop(column, axis=1), onehots], axis=1)
    return dataset

def convert_labels(dataset):
    labels = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "Adelie":0, "Chinstrap":1, "Gentoo":2}
    for label in labels:
        new_label = labels[label]
        dataset.loc[dataset["label"] == label, "label"] = new_label

def split_data(data_set, train_percentage, seed):
    # your code goes here
    np.random.seed(seed)
    train = data_set.sample(frac=train_percentage, random_state=seed)
    test = data_set.drop(train.index)
    train_X, train_Y = train.drop("label", axis=1), train["label"]
    test_X, test_Y = test.drop("label", axis=1), test["label"]
    return train_X, train_Y, test_X, test_Y

# creates a neural network with one hidden layer
def create_network(num_hidden, num_output):
    hidden_layer = tf.keras.layers.Dense(num_hidden, activation='sigmoid')
    output_layer = tf.keras.layers.Dense(1)
    all_layers = [hidden_layer, output_layer]
    network = tf.keras.models.Sequential(all_layers)
    return network

# trains a neural network with given training data
def train_network(network: tf.keras.models.Sequential, training_X, training_y, learning_rate, is_regression):
    if training_X.shape[0] == 0:
        print("Training data is empty.")
        return
    # create the algorithm that learns the weight of the network (with a learning rate of 0.0001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # create the loss function function that tells optimizer how much error it has in its predictions
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Use BinaryCrossentropy for binary classification
    # prepare the network for training
    network.compile(optimizer=optimizer, loss=loss_function, metrics=["mean_absolute_error" if is_regression else "accuracy"])
    # create a logger to save the training details to file
    csv_logger = tf.keras.callbacks.CSVLogger('logger.csv')
    # train the network for 200 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.1, epochs=250, callbacks=[csv_logger])

def predict(network, testing_X, testing_y, is_regression):
        _, performance = network.evaluate(testing_X, testing_y)
        return performance

def main(learning_rate=None, num_neurons=None, train_percentage=None, random_seed=None):
    # Get command line arguments
    if not learning_rate:
        learning_rate = float(sys.argv[1])
        num_neurons = int(sys.argv[2])
        train_percentage = float(sys.argv[3])
        random_seed = int(sys.argv[4])

    dataset = pd.read_csv("data/in/combined_data_with_averages.csv")
    dataset.fillna(0, inplace=True)
    dataset = scale_dataset(dataset)
    num_unique_labels = len(dataset["label"].unique())
    print(dataset.head())

    training_X, training_y, testing_X, testing_y = split_data(dataset, train_percentage, random_seed)
    
    training_X = tf.convert_to_tensor(training_X, dtype=tf.float32)
    training_y = tf.convert_to_tensor(training_y, dtype=tf.float32)
    testing_X = tf.convert_to_tensor(testing_X, dtype=tf.float32)
    testing_y = tf.convert_to_tensor(testing_y, dtype=tf.float32)
    network = create_network(num_neurons, num_unique_labels)

    train_network(network, training_X, training_y, learning_rate, False)
    result = predict(network, testing_X, testing_y, False)
    with open("results.csv", "a") as f:
        f.write(f"{learning_rate},{num_neurons},{result}\n")

if __name__ == "__main__":
    main()