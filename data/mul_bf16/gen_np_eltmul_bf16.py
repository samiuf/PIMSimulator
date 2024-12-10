import numpy as np

def float32_to_bfloat16(arr):
    """Convert float32 array to simulated bfloat16 by truncating the mantissa."""
    view = arr.view(np.uint32)
    view = view & 0xFFFF0000  # Zero out the lower 16 bits of the mantissa
    return view.view(np.float32)

DIM_IN = 1024 * 1024

np.set_printoptions(precision=20)
np.random.seed(1113)
data_in1 = np.random.rand(DIM_IN).astype(np.float32)
data_in2 = np.random.rand(DIM_IN).astype(np.float32)
data_out = np.zeros(DIM_IN).astype(np.float32)

data_in1 = float32_to_bfloat16(data_in1)
data_in2 = float32_to_bfloat16(data_in2)
data_out = float32_to_bfloat16(data_out)

data_out = data_in1 * data_in2

data_out = float32_to_bfloat16(data_out)

np.save("eltmul_input0_" + str(DIM_IN), data_in1)
np.save("eltmul_input1_" + str(DIM_IN), data_in2)
np.save("eltmul_output_" + str(DIM_IN), data_out)

print(data_in1)
print(data_in2)
print(data_out)
