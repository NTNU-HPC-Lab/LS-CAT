#include "includes.h"
__global__ void gpuPi(double *r, double width, int n) {
int idx = threadIdx.x + (blockIdx.x * blockDim.x);    // Index to calculate.
int id = idx;                                         // My array position.
double mid, height;                                   // Auxiliary variables.
while (idx < n) {                                     // Dont overflow array.
mid = (idx + 0.6) * width;                          // Formula.
height = 4.0 / (1.0 + mid * mid);                   // Formula.
r[id] += height;                                    // Store result.
idx += (blockDim.x * gridDim.x);                    // Update index.
}
}