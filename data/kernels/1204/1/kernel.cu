#include "includes.h"
__global__ void kernel(float *array, int size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size) {
array[index] += 1.f;
//if (index == 0)
//  printf("### array[%d] = %f\tArray size: %d\n", index, array[index], size);
}
}