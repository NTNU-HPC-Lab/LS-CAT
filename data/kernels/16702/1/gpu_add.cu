#include "includes.h"
__global__ void gpu_add(int* big_set_numbers, const int big_set_count, int* tiny_set_numbers,const int tiny_set_count)				// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < big_set_count)
{
int total = big_set_numbers[id];
for (int i = 0; i < tiny_set_count; i++)
{
total += tiny_set_numbers[i];
}

big_set_numbers[id] *= total;
}
}