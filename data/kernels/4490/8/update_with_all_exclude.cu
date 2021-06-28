#include "includes.h"
__global__ void update_with_all_exclude(int *clause_output, int *all_exclude)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

// Initialize clause output
for (int j = index; j < CLAUSES; j += stride) {
if (all_exclude[j] == 1) {
clause_output[j] = 0;
}
}
}