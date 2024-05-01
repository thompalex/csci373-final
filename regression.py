import pandas
import sys
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def load_and_split_data(file_path, training_percentage, random_seed):
    data = pandas.read_csv(file_path)
    train_set, test_set = train_test_split(data, test_size=1-training_percentage, random_state=random_seed)
    return train_set, test_set

def scale_dataset(dataset):
    dataset_scaled = dataset.copy()
    for column in dataset_scaled:
        if column.lower() == "label":
            continue
        if min(dataset_scaled[column]) == max(dataset_scaled[column]):
            dataset_scaled[column] = 0
        else:
            dataset_scaled[column] = (dataset_scaled[column] - min(dataset_scaled[column])) / (max(dataset_scaled[column]) - min(dataset_scaled[column]))
    return dataset_scaled

def initialize_models():
    models = {
        "linear": LinearRegression(),
        "LASSO": Lasso(),
        "ridge": Ridge(),
        "svm_poly2": SVR(kernel='poly', degree=2),
        "svm_poly3": SVR(kernel='poly', degree=3),
        "svm_poly4": SVR(kernel='poly', degree=4),
        "svm_rbf": SVR(kernel='rbf'),
        "tree": DecisionTreeRegressor()
    }
    return models

def one_hot_encoder(dataset):
    for column in dataset:
        if column.lower() == "label" or dataset[column].dtype != "object":
            continue
        onehots = pandas.get_dummies(dataset[column], column, drop_first=True, dtype=int)
        dataset = pandas.concat([dataset.drop(column, axis=1), onehots], axis=1)
    return dataset

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        predictions = model.predict(X_test)  # Predict on test data
        mae = mean_absolute_error(y_test, predictions)  # Calculate MAE
        results.append((name, mae))  # Store results
    return results

def create_and_save_bar(results, dataset_name, minmaxusage):
    # Convert results to DataFrame
    results_df = pandas.DataFrame(results, columns=['Model', 'MAE'])

    # Create the bar chart
    bar_chart = (
        ggplot(results_df, aes(x='Model', y='MAE', fill='Model')) +
        geom_col(position="dodge") +  # Use geom_col for bar charts; 'position=dodge' might be omitted if not needed
        ylim(0, max(results_df['MAE']) * 1.1) +  # Dynamic Y-limit based on max MAE value
        labs(
            title=f'MAE for {dataset_name} {"with Rescaling" if minmaxusage == "true" else "without Rescaling"}',
            x='Model',
            y='Mean Absolute Error'
        ) 
    )
    filename = f"{dataset_name}_{'rescaled' if minmaxusage == 'true' else 'original'}_mae_bar.png"
    bar_chart.save(filename=filename)
    print(f"Bar chart saved as {filename}")

def create_and_save_line(results, dataset_name, minmaxusage):
    df = pandas.DataFrame(results)
    line_chart = (
        ggplot(df, aes('Training Percentage', 'MAE', color='Model')) +
        geom_line() +
        labs(title='Model Performance Across Training Percentages',
             x='Training Percentage (%)',
             y='Mean Absolute Error')
    )
    filename = f"{dataset_name}_{'rescaled' if minmaxusage == 'true' else 'original'}_mae_line.png"
    line_chart.save(filename=f"charts/{filename}")
    print(f"Line chart saved as {filename}")

if __name__ == "__main__":
    dataset_filename = ('data/in/combined_data_with_averages.csv')
    split_ratio = float(sys.argv[1])
    random_seed = int(sys.argv[2])
    minmaxusage = sys.argv[3]

    print("Loading data...")
    dataset = pandas.read_csv(dataset_filename)
    print(f"Dataset loaded from {dataset_filename}")

    print(dataset.head())

    print("Applying one-hot encoding...")
    dataset = one_hot_encoder(dataset)
    print("One-hot encoding completed.")

    print(dataset.head())

    if minmaxusage == "true":
        print("Applying min-max scaling...")
        dataset = scale_dataset(dataset)
        print("Scaling completed.")
        print(dataset.head())


    X = dataset.drop('label', axis=1)
    y = dataset['label']


    print(f"Splitting dataset with training ratio {split_ratio} and seed {random_seed}...")
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=random_seed)
    print("Data splitting completed.")

    print("Initializing and training models...")
    # Initialize and train models
    models = initialize_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    print("Models trained and evaluated.")

    create_and_save_bar(results, dataset_filename[:-4], minmaxusage)

    # Output results to CSV
    output_filename = f"regression_results_{int(split_ratio * 100)}p_{random_seed}"
    if minmaxusage == "true":
        output_filename += "_rescaled"
    output_filename += ".csv"

    print(f"Writing results to {output_filename}...")
    with open(output_filename, 'w') as file:
        file.write("Model,MAE\n")
        for model_name, mae in results:
            file.write(f"{model_name},{mae}\n")

    print(f"Results written to {output_filename}")


