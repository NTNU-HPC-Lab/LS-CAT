#include "includes.h"
__global__ void initialize_clause_output_predict(int *clause_output, int *all_exclude)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

// Initialize clause output
for (int j = index; j < CLAUSES; j += stride) {
clause_output[j] = 1;
all_exclude[j] = 1;
}
}