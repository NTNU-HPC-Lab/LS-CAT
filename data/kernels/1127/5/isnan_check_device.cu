#include "includes.h"
__global__ void isnan_check_device(double *array, int size, bool *check) {
//
//  Description: Check for nan in array.

int idx = threadIdx.x + blockDim.x * blockIdx.x;

if (idx < size && ::isnan(array[idx])) {
*check = true;
}
}