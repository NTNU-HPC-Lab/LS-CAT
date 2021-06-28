#include "includes.h"
__global__ void LreluBackward(float* srcDiff, float* dstDiff, float* srcData, int data_size)
{
int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
int num_threads = blockDim.x * gridDim.x;

for(int i = 0; i < data_size; i += num_threads)
{
int index = i + thread_index;
if(index < data_size)
{
dstDiff[index] = srcDiff[index] * ((srcData[index] > 0) + (srcData[index] <= 0) * 0.01);
}
}

}