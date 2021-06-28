#include "includes.h"
__global__ void dropout_train(float* data, float* outputPtr, int size, float probability)
{
int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
int num_threads = blockDim.x * gridDim.x;
for(int  i = 0; i < size; i += num_threads)
{
int index = i + thread_index;
if(index < size)
{
if(outputPtr[index] < probability)
data[index] = 0;
}
}
}