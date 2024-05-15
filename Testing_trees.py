import matplotlib.pyplot as plt
import trees

training_percentages = [0.2, 0.4, 0.6, 0.75]
dataset = "data/in/combined_data_with_hashes_and_averages.csv"


results = []
for training_percentage in training_percentages:
    accuracy = trees.main(dataset, 1, training_percentage, 1234)
    results.append(accuracy)
    # Plot the line chart
plt.plot(training_percentages, results, label=dataset[:-4])

plt.xlabel("Training Percentages")
plt.ylabel("Accuracy")
plt.legend()

# Save the line plot as a PNG file
plt.savefig("trees_test.png", format="png")

# Show the plot
plt.show()