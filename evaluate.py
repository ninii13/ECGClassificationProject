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
  class_correct = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
  class_total = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

  confusion = np.zeros((5,5), dtype=int)
  inf_time = 0.0
  for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
    start_time = time.time()
    # Inference
    outputs = model(inputs)
    inf_time += time.time() - start_time
    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    #Calculate accuracy
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum().item()
    
    #Per class accuracy
    for i in range(len(targets)):
      label = targets[i].item()
      class_total[label] += 1
      if outputs[i].item() == label:
        class_correct[label] += 1

    for t, p in zip(targets, outputs):
      confusion[t.item(), p.item()] += 1

  print("Confusion Matrix: \n", confusion)
  class_accuracy = {}
  
  for cls in class_correct.keys():
    if class_total[cls] > 0:
      class_accuracy[cls] = class_correct[cls] / class_total[cls] * 100
    else:
      class_accuracy[cls] = None
    
  print(f"Average Inference Time Per Batch: {(inf_time / len(dataloader)) * 1000} ms")
  print(f"Accuracy: {(num_correct / num_samples * 100)}%")
  print("Correct: ",class_correct)
  print("Total: ",class_total)
  print(class_accuracy)
