import csv

# Open the csv file
with open('MedQAEvaluated.csv', "r", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = []
    for row in reader:
        data.append(row)

# Do the confusion matrix between column 2 and 3
TP = 0
FP = 0
TN = 0
FN = 0

for row in data:
    if row[3] == "True":
        if row[2] == "True":
            TP += 1
        else:
            FP += 1
    else:
        if row[2] == "True":
            FN += 1
        else:
            TN += 1

# Compute the metrics
accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# Print the confusion matrix using skleanr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
y_true = []
y_pred = []
for row in data:
    y_true.append(row[2])
    y_pred.append(row[3])
disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
disp.plot()
plt.show()

# Print the wrongly classified samples
print("Wrongly classified samples:")
for row in data:
    if row[2] != row[3]:
        print(row[0])
        print(row[1])
        print()