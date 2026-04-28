import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
):
    model.eval()

    num_samples = 0
    num_correct = 0
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    class_total   = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    confusion = np.zeros((5,5), dtype=int)
    inf_time = 0.0

    # --- NEW: store for AUC ---
    all_targets = []
    all_probs   = []

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):

        start_time = time.time()
        logits = model(inputs)
        inf_time += time.time() - start_time

        # --- NEW: probability scores for AUC ---
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

        # Predicted class
        preds = logits.argmax(dim=1)

        # Accuracy
        num_samples += targets.size(0)
        num_correct += (preds == targets).sum().item()

        # Update per-class stats
        for i in range(len(targets)):
            label = targets[i].item()
            class_total[label] += 1
            if preds[i].item() == label:
                class_correct[label] += 1

        # Confusion matrix
        for t, p in zip(targets, preds):
            confusion[t.item(), p.item()] += 1

    # --- Stack stored AUC data ---
    all_probs   = torch.cat(all_probs, dim=0).numpy()  # shape (N,5)
    all_targets = torch.cat(all_targets, dim=0).numpy() # shape (N,)

    # Compute class accuracy %
    class_accuracy = {
        cls: (class_correct[cls] / class_total[cls] * 100) if class_total[cls] > 0 else None
        for cls in class_correct.keys()
    }

    print("Confusion Matrix: \n", confusion)
    print(f"Average Inference Time Per Batch: {(inf_time / len(dataloader)) * 1000} ms")
    print(f"Accuracy: {(num_correct / num_samples * 100)}%")
    print("Correct: ", class_correct)
    print("Total: ", class_total)
    print(class_accuracy)

    # --- RETURN INFO FOR AUC ---
    return all_targets, all_probs, confusion

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from model import ConvolutionalNetwork
from torch.utils.data import DataLoader
from datasets import SEED, test_dataset
BATCH_SIZE = 64
input_length = 221
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = ConvolutionalNetwork(input_length)
model.load_state_dict(torch.load(f"{SEED}_norelu.pth", map_location=torch.device('cpu')))

# After evaluate():
y_true, y_prob, cm = evaluate(model, test_loader)

num_classes = 5
classes = ["N", "V", "S", "F", "Q"]

# binarize (N samples × 5 classes)
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.3f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC–AUC (5 Classes)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
