#include "includes.h"



__global__ void gpu_add(int* gpu_numbers, const int numberCount)				// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < numberCount)
gpu_numbers[id] *= 2;

}