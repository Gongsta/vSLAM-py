from numba import cuda

@cuda.jit
def ComputeDisparityToDepthCUDA(disparity_map, depth_map, width, height, fx, baseline):
    col = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    row = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if col < width and row < height:
        # idx = row * width + col
        if disparity_map[row, col] == 0.0:
            depth_map[row, col] = 0.0
        else:
            depth_map[row, col] = fx * baseline / disparity_map[row, col]
            depth_map[row, col] = min(10.0, depth_map[row, col])
            depth_map[row, col] = max(0.0, depth_map[row, col])

def ComputeDisparityToDepth(disparity_map, depth_map, width, height, fx, baseline):
    grid_dim = (width // 32 + 1, height // 32 + 1, 1)
    block_dim = (32, 32, 1)
    ComputeDisparityToDepthCUDA[grid_dim, block_dim](disparity_map, depth_map, width, height, fx, baseline)
