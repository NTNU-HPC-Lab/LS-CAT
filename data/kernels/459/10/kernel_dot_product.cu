#include "includes.h"
__global__ void kernel_dot_product(const double * vec1, const double * vec2, int             numElements, double       * answer)
{
extern __shared__ double products[]; // one element per thread

int i = threadIdx.x; // numElements assumed to fit into one block
products[i] = vec1[i] * vec2[i];

__syncthreads();

if (i == 0) {
double sum = 0;
for (int j = 0; j < numElements; ++j) {
sum += products[j];
}
*answer = sum;
}
}