import pandas as pd
import numpy as np
import pickle as pkl
import util

TRAIN_PATH = "train_gn.csv"
PARAMETERS_PATH = "parameters_gn.pkl"


def show_array(category_label, array, labels):
    print(f"\t{category_label} -> ", end="")
    for i in range(len(array)):
        print(f"{labels[i]}:{array[i]: >7.4f}     ", end="")
    print()


train_df = pd.read_csv(TRAIN_PATH)
X_df = train_df.iloc[:, :-1]
Y_df = train_df.iloc[:, -1]

n = len(X_df)
d = len(X_df.columns)
print(f"Read {n} samples with {d} attributes from {TRAIN_PATH}")
labels = np.unique(Y_df)
labels_mean = np.zeros((len(labels), d))
labels_stds = np.zeros((len(labels), d))

for i in range(len(labels)):
    label = labels[i]
    labels_mean[i] = X_df[Y_df == label].mean()
    labels_stds[i] = X_df[Y_df == label].std()

# compute the prior probabilities of each class
class_priors = np.zeros(len(labels))
for i in range(len(labels)):
    label = labels[i]
    class_priors[i] = np.sum(Y_df == label) / n

# Print calculated prior percentange
print("Priors:")
for i in range(len(labels)):
    print(f"\t{labels[i]}: {class_priors[i]*100:.1f}%")

# Print calculated mean and standard deviation
for i in range(len(labels)):
    print(f"{labels[i]}:")
    show_array("Means", labels_mean[i], X_df.columns)
    show_array("Stdvs", labels_stds[i], X_df.columns)

# Save the parameters
parameters = {
    "labels": labels,
    "labels_mean": labels_mean,
    "labels_stds": labels_stds,
    "class_priors": class_priors,
}
with open(PARAMETERS_PATH, "wb") as f:
    pkl.dump(parameters, f)
print(f"Wrote parameters to {PARAMETERS_PATH}")
