import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import ConvolutionalNetwork
from datasets import SEED, test_dataset, train_dataset
from evaluate import evaluate

TEST_SIZE = 0.1
BATCH_SIZE = 64

torch.manual_seed(SEED)
input_length = 221
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE)
test_loader2 = DataLoader(test_dataset,batch_size=1)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
# Load model
weights = torch.load(f"{SEED}_norelu.pth", map_location=device)
# for name, param in weights.items():
#     print(f"{name} layer: {param}")
model = ConvolutionalNetwork(input_length)
model.load_state_dict(weights)
print("Model loaded with weights from: ", f"{SEED}_norelu.pth")
# for name, param in model.state_dict().items():
#     print(f"{name} layer: {param}")
# state = torch.load("model_weights3.pth")
# missing, unexpected = model.load_state_dict(state, strict=False)
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)
model = model.to(device)
print("=================== Original Model Evaluation ===================")
model.eval()
evaluate(model, test_loader)

# Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        label = self.y[idx]
        label = torch.tensor(label, dtype=torch.long)
        return data, label

###Calibrate model
# model quantization
class CalibrateConvolutionalNetwork(ConvolutionalNetwork):
    def __init__(self, input_length):
        super().__init__(input_length) #run parent init

        # Create observers for each layer
        self.obs_in = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.obs_conv1 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.obs_conv2 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.obs_conv3 = HistogramObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.obs_fc1 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    def forward(self, X):
        # Observe input
        self.obs_in(X)
        X = self.conv1(X)
        self.obs_conv1(X)
        X = self.conv2(X)
        self.obs_conv2(X)
        X = self.conv3(X)
        self.obs_conv3(X)
        X = F.max_pool1d(X, 8, stride=8)
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        self.obs_fc1(X)
        return X

calib_model = CalibrateConvolutionalNetwork(input_length)
calib_model.load_state_dict(weights, strict=False)
calib_model = calib_model.to(device)

calib_model.eval()
with torch.no_grad():
    for inputs, __ in tqdm(test_loader, desc="calibrating", leave=False):
        inputs = inputs.to(device)
        __ = calib_model(inputs)

### Extract activation quantization parameters
def calc_qparams(observer):
    scale, zp = observer.calculate_qparams()
    return scale.item(), zp.item()

activation_stats = {
    'input': calc_qparams(calib_model.obs_in),
    'conv1': calc_qparams(calib_model.obs_conv1),
    'conv2': calc_qparams(calib_model.obs_conv2),
    'conv3': calc_qparams(calib_model.obs_conv3),
    'fc1': calc_qparams(calib_model.obs_fc1),
}

print("=================== Activation Quantization Parameters ===================")
for name, qparams in activation_stats.items():
    scale = qparams[0]
    zp = qparams[1]
    print(f"{name} -> scale: {scale}, zp: {zp}")

# def quantize_tensor(tensor, scale, zero_point):
#     return torch.clamp(torch.round(tensor/scale + zero_point), 0, 255).to(torch.uint8)
# def dequantize_tensor(q_tensor, scale, zero_point):
#     return scale * (q_tensor.float() - zero_point)
# weight and bias quantization
q_weights = {}
q_bias = {}
weight_qparams = {}
#weight
for name, param in weights.items():
    if 'weight' in name:
        observer2 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        observer2(param)
        scale, zero_point = observer2.calculate_qparams()
        #print(name, observer2.min_val, observer2.max_val)
        #print("Min: ", param.min().item(), " Max: ", param.max().item())
        weight_qparams[name] = (scale.item(), zero_point.item())
        #print(f"Weights - {name} Scale:", scale.item(), "Zero Point:", zero_point.item())
        q_param = torch.quantize_per_tensor(param, scale, zero_point, dtype=torch.qint8)
        q_weights[name] = q_param.int_repr().int()
        #print(f"{name} weights: {q_weights[name].min()} to {q_weights[name].max()}")
#bias
for name, qparam in activation_stats.items():
    if 'input' in name:
        input_scale = qparam[0]
        continue
    bias_name = name + '.base_conv.bias' if 'conv' in name else name + '.bias'
    weight_name = name + '.base_conv.weight' if 'conv' in name else name + '.weight'
    float_bias = weights[bias_name]
    bias_scale = input_scale * weight_qparams[weight_name][0]
    q_bias[bias_name] = torch.quantize_per_tensor(float_bias, bias_scale, 0, dtype=torch.qint32)
    weight_qparams[bias_name] = (bias_scale, 0)
    q_weights[bias_name] = q_bias[bias_name].int_repr().int()
    input_scale = qparam[0]

qparams = {
    'input': {'input_scale': activation_stats['input'][0], 'input_zp': activation_stats['input'][1]},
    'conv1': {
        'weight_scale': weight_qparams['conv1.base_conv.weight'][0],
        'weight_zp': weight_qparams['conv1.base_conv.weight'][1],
        'bias_scale': weight_qparams['conv1.base_conv.bias'][0],
        'output_scale': activation_stats['conv1'][0],
        'output_zp': activation_stats['conv1'][1],
    },
    'conv2': {
        'weight_scale': weight_qparams['conv2.base_conv.weight'][0],
        'weight_zp': weight_qparams['conv2.base_conv.weight'][1],
        'bias_scale': weight_qparams['conv2.base_conv.bias'][0],
        'output_scale': activation_stats['conv2'][0],
        'output_zp': activation_stats['conv2'][1],
    },
    'conv3':{
        'weight_scale': weight_qparams['conv3.base_conv.weight'][0],
        'weight_zp': weight_qparams['conv3.base_conv.weight'][1],
        'bias_scale': weight_qparams['conv3.base_conv.bias'][0],
        'output_scale': activation_stats['conv3'][0],
        'output_zp': activation_stats['conv3'][1],
    },
    'fc1':{
        'weight_scale': weight_qparams['fc1.weight'][0],
        'weight_zp': weight_qparams['fc1.weight'][1],
        'bias_scale': weight_qparams['fc1.bias'][0],
        'output_scale': activation_stats['fc1'][0],
        'output_zp': activation_stats['fc1'][1],
    }
}

### Convert to quantized model
class QuantizedConvolutionalNetwork():
    def __init__(self, q_weights, qparams):
        self.qparams = qparams
        self.q_weights = q_weights

    def quantized_conv_layer(
            self, x,
            in_channels,
            weights,
            q_bias,
            input_zp,
            weight_zp
        ):
        # Shared kernel logic
        if weights.size(1) == 1 and in_channels > 1:
            weights = weights.repeat(1, in_channels, 1)

        x_off = (x.float())
        ###
        weights_off = (weights.float())
        # print("x_off: ", x_off)
        # print("weights_off: ", weights_off)
        ###
        x = F.conv1d(x_off, weights_off, q_bias.float(), stride=1)
        return x

    def quantized_linear_layer(
            self, x,
            weights,
            q_bias,
            input_zp,
            weight_zp
        ):
        x_off = (x.float())

        weights_off = (weights.float())
        #print("input: ", x_off, "weight: ", weights_off)
        x = F.linear(x_off, weights_off, q_bias.float())
        #print("output: ", x)
        return x

    def rescale_hw(self, x, divisor):
        x = torch.trunc(x.float() / divisor).to(torch.int32)
        return torch.clamp(x, -128, 127).to(torch.int8)

    def rescale(self, x, input_scale, weight_scale, output_scale):
        x = x * (input_scale * weight_scale / output_scale)
        return torch.clamp(torch.round(x), -128, 127).to(torch.int8)

    def quant_stub(self, x, scale, zero_point):
        x = torch.clamp(torch.round(x/scale + zero_point), -128, 127)
        return x.to(torch.int8)

    def dequant_stub(self, x, scale, zero_point):
        return scale * (x.float() - zero_point)

    def forward(self, X):
        #Convolutional Layers
        X = self.quant_stub(X, qparams['input']['input_scale'], qparams['input']['input_zp'])
        #print("after quant_stub", X.dtype, X.min().item(), X.max().item())
        X = self.quantized_conv_layer( X, 1, q_weights['conv1.base_conv.weight'], q_weights['conv1.base_conv.bias'], qparams['input']['input_zp'], qparams['conv1']['weight_zp'])
        # print("Convolution 1 Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        #X = self.rescale(X, qparams['input']['input_scale'], qparams['conv1']['weight_scale'], qparams['conv1']['output_scale'])
        X = self.rescale_hw(X, 281)
        # print("Convolution 1 Quantized Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        X = self.quantized_conv_layer( X, 2, q_weights['conv2.base_conv.weight'], q_weights['conv2.base_conv.bias'], qparams['conv1']['output_zp'], qparams['conv2']['weight_zp'])
        # print("Convolution 2 Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        # print("activation output channel 3: ", X[0,2,0:20])
        # print("activation output channel 4: ", X[0,3,0:20])
        #X = self.rescale(X, qparams['conv1']['output_scale'], qparams['conv2']['weight_scale'], qparams['conv2']['output_scale'])
        X = self.rescale_hw(X, 491)
        # print("Convolution 2 Quantized Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        # print("activation output channel 3: ", X[0,2,0:20])
        # print("activation output channel 4: ", X[0,3,0:20])
        X = self.quantized_conv_layer( X, 4, q_weights['conv3.base_conv.weight'], q_weights['conv3.base_conv.bias'], qparams['conv2']['output_zp'], qparams['conv3']['weight_zp'])
        # print("Convolution 3 Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        #X = self.rescale(X, qparams['conv2']['output_scale'], qparams['conv3']['weight_scale'], qparams['conv3']['output_scale'])
        X = self.rescale_hw(X, 1247)
        # print("Convolution 3 Quantized Output Results")
        # print("activation output channel 1: ", X[0,0,0:20])
        # print("activation output channel 2: ", X[0,1,0:20])
        # Pooling Layer
        X = F.max_pool1d(X.float(), 8, stride=8)
        X = torch.clamp(torch.round(X), -127, 128)
        # Flatten
        X = X.view(X.size(0), -1)
        # Fully Connected Layer
        X = self.quantized_linear_layer(X, q_weights['fc1.weight'], q_weights['fc1.bias'], qparams['conv3']['output_zp'], qparams['fc1']['weight_zp'])
        #print("output: ", X)
        #X = self.dequant_stub(X, qparams['conv3']['output_scale']*qparams['fc1']['weight_scale'], qparams['fc1']['output_zp'])
        return X
q_model = QuantizedConvolutionalNetwork(q_weights, qparams)

# #Example pass
# test_index = 10
# example_input = test_loader.dataset[test_index][0].unsqueeze(0)
# example_output = test_loader.dataset[test_index][1]
# q_output = q_model.forward(example_input)
# #Compare with original
# model.eval()
# with torch.no_grad():
#     model_output = model(example_input.to(device))
# print("Original model output: ", model_output, "Predicted class: ", model_output.argmax(dim=1).item())
# print("Quantized model output: ", q_output, "Predicted class: ", q_output.argmax(dim=1).item())
# print("Correct class: ", example_output)
# quantized_signals = torch.clamp(torch.round(example_input/qparams['input']['input_scale'] + qparams['input']['input_zp']), -128, 127).to(torch.int8)

# with open(f"input.txt","w") as file:
#     file.write("vector<int> input = {\n")
#     file.write(",".join(str(x.item()) for x in quantized_signals[0,0,:]) + "\n")
#     file.write(" };\n\n")
#Evaluate
num_samples = 0
num_correct = 0
class_correct = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
class_total = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

a,b,c,d,e = True,True,True,True,True
index = 1
test_correct = []
q = []

confusion = np.zeros((5,5), dtype=int)

for inputs, targets in tqdm(test_loader, desc="eval", leave=False):
    # Inference
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = q_model.forward(inputs)

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

print("=================== Quantized Model Evaluation ===================")
print("Confusion Matrix: \n", confusion)
class_accuracy = {}

for cls in class_correct.keys():
    if class_total[cls] > 0:
        class_accuracy[cls] = class_correct[cls] / class_total[cls] * 100
    else:
        class_accuracy[cls] = None

print(f"Accuracy: {(num_correct / num_samples * 100)}%")
print("Correct: ",class_correct)
print("Total: ",class_total)
print(class_accuracy)
label_map = {'N':0, 'VEB':1, 'SVEB':2, 'F':3, 'Q':4}


# ----------------------------
# Confusion matrix
# ----------------------------
cm_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Normalized Confusion Matrix (per class)")
plt.show()

# def save_qweights(q_weights, filename="ptq_quantized_weights.txt"):
#     with open(filename, "w") as f:
#         # Sort base_conv, weight, bias by layer
#         layers = {}
#         for key, tensor in q_weights.items():
#             layer = key.split('.')[0]  # "conv1.base_conv.weight" → "conv1"
#             if layer not in layers:
#                 layers[layer] = {}
#             if "weight" in key:
#                 layers[layer]["weight"] = tensor
#             elif "bias" in key:
#                 layers[layer]["bias"] = tensor
#         for layer in sorted(layers.keys()):

#             # WRITE WEIGHT
#             if "weight" in layers[layer]:
#                 arr_w = layers[layer]["weight"].cpu().numpy().astype(int)
#                 arr_w_str = np.array2string(arr_w, separator=' ', max_line_width=200)
#                 f.write(f"Layer: {layer}.weight\n")
#                 f.write(arr_w_str + "\n\n")

#             # WRITE BIAS
#             if "bias" in layers[layer]:
#                 arr_b = layers[layer]["bias"].cpu().numpy().astype(int)
#                 arr_b_str = np.array2string(arr_b, separator=' ', max_line_width=200)
#                 f.write(f"Layer: {layer}.bias\n")
#                 f.write(arr_b_str + "\n")

#             f.write("\n")  # blank line between layers
#         # for key, tensor in q_weights.items():
#         #     arr = tensor.cpu().numpy().astype(int)

#         #     # Write layer name
#         #     f.write(f"Layer: {key}\n")

#         #     # Convert numpy array to string WITHOUT commas
#         #     arr_str = np.array2string(arr, separator=' ', max_line_width=200)

#         #     # Write the array
#         #     f.write(arr_str + "\n\n")
#         conv1_rescale = qparams['input']['input_scale'] * qparams['conv1']['weight_scale'] / qparams['conv1']['output_scale']
#         conv2_rescale = qparams['conv1']['output_scale'] * qparams['conv2']['weight_scale'] / qparams['conv2']['output_scale']
#         conv3_rescale = qparams['conv2']['output_scale'] * qparams['conv3']['weight_scale'] / qparams['conv3']['output_scale']
#         f.write("=== Requantization parameters between layers ===\n")
#         f.write(f"conv1: scale={math.ceil(conv1_rescale * 1048576)}, zp={qparams['conv1']['output_zp']}\n")
#         f.write(f"conv2: scale={math.ceil(conv2_rescale * 1048576)}, zp={qparams['conv2']['output_zp']}\n")
#         f.write(f"conv3: scale={math.ceil(conv3_rescale * 1048576)}, zp={qparams['conv3']['output_zp']}\n")
#     print(f"Saved formatted quantized weights to: {filename}")

#save_qweights(q_weights, filename="ptq_quantized_weights1.txt")
