#include "includes.h"
__global__ void __add__(int *array, int *size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx > *size) return;

int temp = 0;
int before = (idx + 1) % *size;
int after = idx - 1;
if (after < 0) after = *size - 1;


temp += array[idx];
temp += array[before];
temp += array[after];

__syncthreads(); // Barrera...
array[idx] = temp;
}