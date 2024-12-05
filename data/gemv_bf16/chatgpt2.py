import torch
import numpy as np

# Parameters
BATCH = 1
REAL_DIM_IN = 1024
DIM_IN = 1024
DIM_OUT = 4096

torch.manual_seed(1113)

# Step 1: Initialize Input Matrix (bfloat16 in Torch)
batch_in = torch.randn((DIM_IN, BATCH), dtype=torch.bfloat16)
# Zero-out padding
batch_in[REAL_DIM_IN:DIM_IN, :] = 0

# Step 2: Initialize Weight Matrix (bfloat16 in Torch)
data_w = torch.randn((DIM_OUT, DIM_IN), dtype=torch.bfloat16)
data_w = data_w[torch.randperm(data_w.size(0))]  # Shuffle rows

# Step 3: Matrix Multiplication (Torch)
batch_out = torch.matmul(data_w, batch_in)

# Step 4: Manual Matrix Multiplication (Verification)
batch_out2 = torch.zeros((DIM_OUT, BATCH), dtype=torch.bfloat16)
for y in range(DIM_OUT):
    for x in range(DIM_IN):
        batch_out2[y] += data_w[y, x] * batch_in[x, 0]

# Step 5: Convert to NumPy Arrays
# Upcast to float for conversion as NumPy doesn't support bfloat16 directly
batch_in_np = batch_in.half().numpy()
data_w_np = data_w.half().numpy()
batch_out_np = batch_out.half().numpy()
batch_out2_np = batch_out2.half().numpy()

# Step 6: Save Data to Files
np.save("gemv_input_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_in_np)
np.save("gemv_weight_" + str(DIM_OUT) + "x" + str(DIM_IN), data_w_np)
np.save("gemv_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out_np)
np.save("test_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out2_np)

# Step 7: Print Results
print(batch_in_np)
print(batch_out_np)
print(batch_out2_np)
print(batch_in_np.shape)
print(batch_out_np.shape)