Optimizing Deep Neural Network Models for Abnormal ECG Signal Detection Capstone Project (2025)
Project Highlights: Shared Kernel Architecture, SMOTE Sampling Technique, Post-Training Quantization
<img width="537" height="607" alt="orig_model_graph" src="https://github.com/user-attachments/assets/99ed7022-6bc2-49a7-aeb2-f9d26d12b88e" />

Instructions:
1. Download the entire folder from the MITBIH Arrhythmia Database (https://physionet.org/content/mitdb/1.0.0/)
2. The data folder should be named "mitdb" and put in the same directory as this project folder
   
   E.g.
   
       root/
        ├──mitdb/
        └── ECGClassificationProject/
4. Run "preprocess.py", it should generate the processed data files: heartbeats.csv and labels.csv
5. Run "train.py" to start training the model
6. Run "plot_confusion_matrix.py" to visualize results
7. Run "ptq.py" to see quantized model results

Additionally, 

"datasets.py": Splits the data into training, validation, and test datasets

"model.py": Model architecture definition

"evaluate.py": Evaluates model accuracy and generates a simple confusion matrix

These are files that do not need to be run individually.

User Input function is also added for the seed, can be set to any integer or a random number by typing in "random".

操作說明 (Instructions)
1. 下載數據集：從 MIT-BIH 心律不整資料庫 (MITBIH Arrhythmia Database) 下載完整的資料夾。 (網址：https://physionet.org/content/mitdb/1.0.0/)
2. 路徑設定：將下載的數據資料夾命名為 mitdb，並將其與此專案資料夾放在同一個目錄下。

   例如：
   
        根目錄/
        ├── mitdb/
        └── ECGClassificationProject/
4. 數據預處理：執行 preprocess.py。該程式會生成處理後的數據文件：heartbeats.csv 與 labels.csv。
5. 模型訓練：執行 train.py 開始訓練模型。
6. 結果視覺化：執行 plot_confusion_matrix.py 查看訓練結果的視覺化圖表（混淆矩陣）。
7. 模型量化：執行 ptq.py 查看量化後模型 (Post-Training Quantization) 的執行結果。

補充說明

datasets.py: 負責將數據切分為訓練集 (Training)、驗證集 (Validation) 與測試集 (Test)。

model.py: 模型架構定義。

evaluate.py: 評估模型準確度並生成基礎混淆矩陣。

注意：以上三個檔案不需要單獨執行。

隨機種子設定 (Seed)

程式已加入使用者輸入功能，你可以輸入任何整數來固定隨機種子；或者輸入 "random" 來產生隨機數值。
