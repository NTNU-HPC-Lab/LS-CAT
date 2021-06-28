#include "includes.h"
/*
* CudaOperations.cu
*
*  Created on: Feb 6, 2019
*      Author: alexander
*/


__global__ void quickSum(double* energyTempor, int size) {
long long offset = 1;
int wIndex;
while (offset < size * size) {
wIndex = threadIdx.x;
while ((wIndex * 2 + 1) * offset < size * size) {
energyTempor[wIndex * 2 * offset] += energyTempor[(wIndex * 2 + 1)
* offset];
wIndex = wIndex + blockDim.x;
}
offset *= 2;
__syncthreads();
}
}