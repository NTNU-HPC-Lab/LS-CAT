#include "includes.h"
__global__ void operator_sum_h(const float *input1, float *output, const int *input1_shape, int input1_dims, const int *temp_shape, int dim, int dim_stride, int size) {
extern __shared__ int shared[];

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
int *loc = (int *)shared + threadIdx.x * input1_dims;

index2loc(index, temp_shape, input1_dims - 1, loc);
for (int i = input1_dims - 1; i > dim; i--) {
loc[i] = loc[i - 1];
}
loc[dim] = 0;
int base = loc2index(loc, input1_shape, input1_dims);

int length = input1_shape[dim];
double total = 0;
for (int i = 0; i < length; i++) {
total += input1[base + i * dim_stride];
}

output[index] = total;
}
}