#include "includes.h"
__global__ void LreluForward(float* srcData, float* dstData, int data_size)
{
int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
int num_threads = blockDim.x * gridDim.x;
for(int i = 0; i < data_size; i += num_threads)
{
int index = i + thread_index;
if(index < data_size)
{
dstData[index] = srcData[index] > 0 ? srcData[index] : srcData[index] * 0.01;
}
}

}