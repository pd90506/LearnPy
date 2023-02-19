#%%
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# %%
import pandas as pd

data = pd.read_csv(data_file)
print(data)
# %%
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
# %%
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# %%
# Conversion to the tensor format
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
# %%
# scalers
import torch
x = torch.tensor([3.0])
y = torch.tensor([2.0])
x + y, x * y, x / y, x**y

# %%
x = torch.arange(4)
x
# %%
A = torch.arange(20).reshape(5,4)
A
# %%
A.T
# %%
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
# %%
B == B.T
# %%
# Tensors
X = torch.arange(24).reshape(2, 3, 4)
X
# %%
# Basic properties of tensor arithmetic
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
A, A + B
# %%
# Hadamard product
A * B
# %%
# operation with scaler
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
# %%
# reduction
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
# %%
A.shape, A.sum()
# %%
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
# %%
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
# %%
A.sum(axis=[0, 1])
# %%
A.mean(), A.sum() / A.numel()
# %%
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
# %%
# non-reduction sum
sum_A = A.sum(axis=1, keepdims=True)
sum_A

# %%
A / sum_A
# %%
A.cumsum(axis=0)
# %%
# dot product
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y)
# %%
torch.sum(x * y)
# %%
