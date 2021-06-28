#include "includes.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__global__ void render(float *fb, int max_x, int max_y) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;
if((i >= max_x) || (j >= max_y)) return;
int pixel_index = j*max_x*3 + i*3;
fb[pixel_index + 0] = float(i) / max_x;
fb[pixel_index + 1] = float(j) / max_y;
fb[pixel_index + 2] = 0.0;
}