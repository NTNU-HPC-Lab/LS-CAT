#include "includes.h"



#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__global__ void transform(float *input, const float *raw_input, const int width, const int channels)
{
int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
int start_i = thread_id / width - 1;
int start_j = thread_id % width - 1;
int hidden_width = 3 * 3 * channels + 1;
int global_offset = thread_id * hidden_width;

float relu;
for (int c = 0; c < channels; c++) {
int offset = 0;
for (int i = start_i; i < start_i + 3; i++) {
if (i < 0 || i == width)
continue;
for (int j = start_j; j < start_j + 3; j++) {
if (j < 0 || j == width)
continue;
relu = raw_input[(i * width + j) * channels + c];
input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
offset++;
}
}
}
input[(thread_id + 1) * hidden_width - 1] = 1;
}