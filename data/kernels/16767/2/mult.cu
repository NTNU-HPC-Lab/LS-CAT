#include "includes.h"
__global__ void mult(int* results, int* data, int* vec) {
int index = blockIdx.x * blockDim.x  + threadIdx.x;
int result_val = 0;
for(int i = 0; i < cuda_features; i++) {
result_val += vec[i] * data[(index * cuda_features) + i];
}
results[index] = result_val;
}