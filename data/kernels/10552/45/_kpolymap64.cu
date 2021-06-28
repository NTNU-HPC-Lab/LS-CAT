#include "includes.h"
__global__ void _kpolymap64(int n, double *k, double c, double d) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
k[i] = pow(k[i] + c, d);
i += blockDim.x * gridDim.x;
}
}