#include "includes.h"
__global__ void compute_array_square(float* array, float* outArray, int size)
{
int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
int num_threads = blockDim.x * gridDim.x;

for(int i = 0; i < size; i += num_threads)
{
int index = i + thread_index;
if(index < size)
{
outArray[index] = array[index] * array[index];
}
}
}