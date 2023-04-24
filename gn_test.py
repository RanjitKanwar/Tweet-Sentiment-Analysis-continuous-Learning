import pandas as pd
import numpy as np
import pickle as pk
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import csv
import util

TEST_PATH = "test_gn.csv"
CONFUSION_PATH = "confusion_gn.png"
PLOT_PATH = "confidence_gn.png"
PARAMETERS_PATH = "parameters_gn.pkl"

## Your code here

print(f"Read parameters from {PARAMETERS_PATH}")
with open(PARAMETERS_PATH, "rb") as f:
    parameters = pk.load(f)

### Your code here

test_df = pd.read_csv(TEST_PATH)
X = test_df.iloc[:, :-1].to_numpy(dtype=np.float64)
n = X.shape[0]
Y_df = test_df.iloc[:, -1]
print(f"Read {len(test_df)} rows from {TEST_PATH}")

# Gather ground truth and predictions
predicted_values = []
prediction_confidences = []
constant = np.log(2 * np.pi) / 2

# initialize this to skip column name row
skip = 0
# Step through the test.csv file
with open("test_gn.csv", "r") as f:
    reader = csv.reader(f)

    skipped_tweet_count = 0

    prediction = []
    pred_prob = []
    gt = []
    log_of_posteriors = []
    for row in reader:
        # print(row)
        if skip == 0:
            skip += 1
            continue

        # Check to see if the row has two entries
        if len(row) != 5:
            continue

        confidence = []

        for label_index in range(len(parameters["labels"])):
            likelihood = 0
            for i in range(4):
                likelihood_att = (
                    -np.log(parameters["labels_stds"][label_index][i])
                    - constant
                    - (
                        (
                            (float(row[i]) - parameters["labels_mean"][label_index][i])
                            / parameters["labels_stds"][label_index][i]
                        )
                        ** 2
                        / 2
                    )
                )

                likelihood += likelihood_att

            log_posteriors = likelihood + np.log(
                parameters["class_priors"][label_index]
            )
            log_of_posteriors.append(log_posteriors)
            confidence.append(log_posteriors)

        prediction.append(np.argmax(confidence))
        pred_prob.append(max(confidence))
        gt.append(list(parameters["labels"]).index(row[-1]))

n = len(log_of_posteriors) // 5
l = len(parameters["labels"])

log_of_posteriors = np.reshape(log_of_posteriors, (n, l))
pred_prob = np.array(pred_prob)
prediction_probs = (
    np.exp(log_of_posteriors - pred_prob[:, np.newaxis])
    / np.sum(np.exp(log_of_posteriors - pred_prob[:, np.newaxis]), axis=1)[
        :, np.newaxis
    ]
)


print("Here are 10 rows of results:")
for i, value in enumerate(gt):
    if i == 10:
        break
    print(f"\tGT={ parameters['labels'][value] }->", end="")
    for j in range(len(parameters["labels"])):
        print(
            f"\t{parameters['labels'][j]}: {prediction_probs[i, j]*100.0:.1f}%", end=""
        )
    print()


print("\n*** Analysis ***")
diff = np.array(gt) - np.array(prediction)
correct = np.sum(diff == 0)
accuracy = correct / n
print(f"{n} data points analyzed, {correct} correct ({accuracy * 100.0:.1f}% accuracy)")

# Confusion matrix
cm = confusion_matrix(np.array(gt), np.array(prediction))
print("Confusion:\n", cm)

labels = parameters["labels"]
priors = parameters["class_priors"]
best_label_idx = np.argmax(priors)

# Save out a confusion matrix plot
## Your code here
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion_gn.png")
print("Wrote confusion matrix matrix plot to confusion_gn.png")

print("\n*** Making a plot ****")

confidence = np.max(prediction_probs, axis=1)
steps = 32
thresholds = np.linspace(0.2, 1.0, steps)
correct_ratio = np.zeros(steps)
confident_ratio = np.zeros(steps)

for i in range(steps):
    threshold = thresholds[i]
    if np.sum(confidence > threshold) != 0:
        correct_ratio[i] = np.sum(
            (confidence >= threshold) & (np.array(gt) == prediction)
        ) / np.sum(confidence >= threshold)
    else:
        correct_ratio[i] = 1
    confident_ratio[i] = np.sum(confidence >= threshold) / n

fig, ax = plt.subplots()
ax.set_title("Confidence and Accuracy Are Correlated")
ax.set_xlabel("Confidence Threshold")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100.0:.0f}%")
ax.plot(
    thresholds, correct_ratio, "blue", linewidth=0.8, label="Accuracy Above Threshod"
)
ax.plot(
    thresholds,
    confident_ratio,
    "r",
    linestyle="dashed",
    linewidth=0.8,
    label="Test data scoring above threshold",
)
ax.hlines(
    priors[best_label_idx],
    0.2,
    1,
    "blue",
    linestyle="dashed",
    linewidth=0.8,
    label=f"Accuracy Guessing {labels[best_label_idx]}",
)
ax.legend()
fig.savefig("confidence_gn.png")

print(f'Saved to "{PLOT_PATH}".')
