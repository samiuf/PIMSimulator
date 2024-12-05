import numpy as np

# Function to simulate bfloat16 conversion
def float32_to_bfloat16(arr):
    """Convert float32 array to simulated bfloat16 by truncating the mantissa."""
    view = arr.view(np.uint32)
    view = view & 0xFFFF0000  # Zero out the lower 16 bits of the mantissa
    return view.view(np.float32)

# Constants
BATCH = 1
REAL_DIM_IN = 1024
DIM_IN = 1024
DIM_OUT = 4096

np.set_printoptions(precision=20)
np.random.seed(1113)

# Input data
batch_in = np.random.standard_normal(size=(DIM_IN, BATCH)).astype(np.float32)
for i in range(REAL_DIM_IN, DIM_IN):
    for j in range(0, BATCH):
        batch_in[i][j] = 0

# Convert to bfloat16
batch_in = float32_to_bfloat16(batch_in)

# Weight matrix
data_w = np.random.standard_normal(size=(DIM_OUT, DIM_IN)).astype(np.float32)

# Convert weights to bfloat16
data_w = float32_to_bfloat16(data_w)

np.random.shuffle(data_w)

# Output computation
batch_out = np.zeros((DIM_OUT, BATCH), dtype=np.float32)
batch_out = np.matmul(data_w, batch_in)

# Simulate bfloat16 output
batch_out = float32_to_bfloat16(batch_out)

# Manual computation
batch_out2 = np.zeros((DIM_OUT, BATCH), dtype=np.float32)

for y in range(0, DIM_OUT):
    for x in range(0, DIM_IN):
        batch_out2[y] += data_w[y][x] * batch_in[x][0]

# Simulate bfloat16 output
batch_out2 = float32_to_bfloat16(batch_out2)

# Transpose and save
batch_in = batch_in.T.copy()
batch_out = batch_out.T.copy()
batch_out2 = batch_out2.T.copy()

np.save("gemv_input_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_in)
np.save("gemv_weight_" + str(DIM_OUT) + "x" + str(DIM_IN), data_w)
np.save("gemv_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out)
np.save("test_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out2)

# Print statements for verification
print(batch_in)
print(batch_out)
print(batch_out2)
print(batch_in.shape)
print(batch_out.shape)
