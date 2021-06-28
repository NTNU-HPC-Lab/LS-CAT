#include "includes.h"
__global__ void add_constant(int* arr, int k, int arr_size) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < arr_size) {
arr[i] += k;
}
}