#include "includes.h"
__global__ void zero_fill_int(int *d_array, int size) {

int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= size) {
return;
}

d_array[i] = 0;

}