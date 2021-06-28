#include "includes.h"
__global__ void TopForcing(double ppt, double *eff_rain, int size) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < size) {
eff_rain[tid] = ppt;
tid += blockDim.x * gridDim.x;
}
}