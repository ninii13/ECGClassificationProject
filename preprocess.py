import wfdb
from wfdb import processing
#wfdb.dl_database('mitdb', dl_dir='.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import os

# Local folder with mitdb files
folder_path =  os.path.join('..', 'mitdb')
records = [f.split('.')[0] for f in os.listdir(folder_path) if f.endswith('.dat')]

# AAMI mapping: raw annotations → 5 heartbeat classes
symbol_map = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'SVEB', 'a': 'SVEB', 'J': 'SVEB', 'S': 'SVEB',
    'V': 'VEB', 'E': 'VEB',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q', '?': 'Q'
}

#Splice waveform into different heartbeat segments
Symbols = ['N', 'SVEB', 'VEB', 'F', 'Q']
heartbeats = []
labels = []
count = 0
for record_name in records:
  record_path = os.path.join(folder_path, record_name)
  # Read full signal and annotation ONCE
  record = wfdb.rdrecord(record_path)
  try:
        #get heartbeat type annotations
        ecg_ann = wfdb.rdann(record_path, extension='atr')
  except FileNotFoundError:
        print(f"No annotation found for {record_name}, skipping.")
        continue
  # Check if 'MLII' lead exists
  if 'MLII' not in record.sig_name:
        print(f"MLII lead not found in {record_name}, skipping.")
        continue

  marks = ecg_ann.sample #get positions of r peaks (annotation symbols)
  classes = ecg_ann.symbol #get annotation symbols
  lead_idx = record.sig_name.index('MLII')
  signal = record.p_signal[:, lead_idx]
  """
  ###downsample to 128 samples/s = 128 Hz
  original_fs = 360  # original sampling rate (Hz)
  target_fs = 128    # target sampling rate (Hz)
  duration_sec = len(signal) / original_fs  # duration of the signal in seconds
  num_samples_resampled = int(duration_sec * target_fs)  # new number of samples
  signal = resample(signal, num_samples_resampled)
  ###
  """
  sig_len = len(signal)
  start = 0
  end = 0

  #extract hearbeats
  for i in range(len(marks)):
    #filter unwanted beat types
    if classes[i] == '.':
        classes[i] = 'N'
    sym = symbol_map.get(classes[i])
    if sym not in Symbols:
      continue
    start = marks[i] - 110
    end = marks[i] + 111
    if start < 0 or end > sig_len:
      continue  # Skip if out of bounds
    heartbeats.append(signal[int(start):int(end)])
    labels.append(sym)
#     if sym == 'N' and count>10000:
#       continue
#     else:
#       if sym == 'N':
#          count += 1
#       heartbeats.append(signal[int(start):int(end)])
#       labels.append(sym)
      

# Save to CSV
heartbeats = pd.DataFrame(heartbeats)
labels = pd.DataFrame(labels)
heartbeats.to_csv('heartbeats.csv', index=False)
labels.to_csv('labels.csv', index=False)

print("Finished extracting and saving heartbeats and labels!")