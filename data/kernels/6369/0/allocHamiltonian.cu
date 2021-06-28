#include "includes.h"
/*
* CudaOperations.cu
*
*  Created on: Feb 6, 2019
*      Author: alexander
*/


__global__ void allocHamiltonian(float* devMat, float* devSpins, int index, int size, double* energyTempor) {
int i;
int j;

int wIndex = threadIdx.x + blockIdx.x * blockDim.x;
while (wIndex < size * size) {
i = wIndex % size;
j = (int) (wIndex / size);
energyTempor[wIndex] = (double) (devSpins[i + index * size]
* devSpins[j + index * size] * devMat[wIndex]);
wIndex = wIndex + blockDim.x * gridDim.x;
}
}