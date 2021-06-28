#include "includes.h"
__global__ void kernel_euclidean_norm(const double      *vec, int                 numElements, double            *answer)
{
extern __shared__ double square[]; // one element per thread

int i = threadIdx.x; // numElements assumed to fit into one block
square[i] = vec[i] * vec[i];

__syncthreads();

if (i == 0) {
double sum = 0;
for (int j = 0; j < numElements; ++j) {
sum += square[j];
}
*answer = sqrt(sum);
}
}