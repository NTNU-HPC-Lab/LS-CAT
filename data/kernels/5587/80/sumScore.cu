#include "includes.h"
__global__ void sumScore(double *score, int full_size, int half_size)
{
int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = index; i < half_size; i += stride) {
score[i] += (i + half_size < full_size) ? score[i + half_size] : 0;
}
}