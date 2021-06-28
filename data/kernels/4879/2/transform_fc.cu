#include "includes.h"



#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__global__ void transform_fc(float *input, const float *raw_input, const int width, const int channels)
{
int thread_id = threadIdx.x;
int size = width * width;

for (int s = 0; s < size; s++)
input[thread_id * size + s] = raw_input[s * channels + thread_id];
if (thread_id == 0)
input[width * width * channels] = 1;
}