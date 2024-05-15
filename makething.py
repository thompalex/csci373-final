import matplotlib.pyplot as plt

accuracies_dict = {'svm_pol4_scaled': 93.1, 'decision_tree': 93.7, 'random_forest': 94.6, 'neural': 93.7}

# Extract the model names and accuracies
models = list(accuracies_dict.keys())
accuracies = list(accuracies_dict.values())


# Plot the accuracies
plt.ylim(bottom=90, top=96)
plt.bar(models, accuracies, color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()