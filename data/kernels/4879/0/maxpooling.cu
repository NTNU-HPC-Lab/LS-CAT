#include "includes.h"



#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__global__ void maxpooling(float *output, const float *input, const int width, const int channels)
{
int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
int new_width = width / 2;
int i = thread_id / new_width * 2;
int j = thread_id % new_width * 2;
int index = i * width + j;

for (int c = 0; c < channels; c++) {
float max = 0;
if (max < input[index * channels + c])
max = input[index * channels + c];
if (max < input[(index + 1) * channels + c])
max = input[(index + 1) * channels + c];
if (max < input[(index + width) * channels + c])
max = input[(index + width) * channels + c];
if (max < input[(index + width + 1) * channels + c])
max = input[(index + width + 1) * channels + c];
output[thread_id * channels + c] = max;
}
}