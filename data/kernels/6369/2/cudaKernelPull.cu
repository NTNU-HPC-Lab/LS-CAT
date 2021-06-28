#include "includes.h"
/*
* CudaOperations.cu
*
*  Created on: Feb 6, 2019
*      Author: alexander
*/


__global__ void cudaKernelPull(float* mat, float* spins, int size, float* temp, float tempStep, float* meanFieldElements, bool* continueIteration, float minDiff, int* unemptyCells, float linearCoef) {
int blockId = blockIdx.x;
int thrId = threadIdx.x;

do {
// Lessen temperature
if (thrId == 0)
temp[blockId] = temp[blockId] - tempStep;

// Stabilize
do {
__syncthreads();
// By default current iteration is the last one
if (thrId == 0)
continueIteration[blockId] = false;

for (int spinId = 0; spinId < size; ++spinId) {
__syncthreads();

// Transitional value assignment
int wIndex = thrId;
while (wIndex < unemptyCells[spinId * (size + 1)]) {
meanFieldElements[wIndex + blockId * size] =
spins[unemptyCells[spinId * (size + 1) + wIndex + 1]
+ blockId * size]
* mat[spinId * size
+ unemptyCells[spinId * (size + 1)
+ wIndex + 1]];
// BEWARE: Matrix is symmetrical!
wIndex = wIndex + blockDim.x;
}
__syncthreads();

// Parallelized mean-field computation
long long offset = 1;
while (offset < unemptyCells[spinId * (size + 1)]) {
wIndex = thrId;
while ((wIndex * 2 + 1) * offset
< unemptyCells[spinId * (size + 1)]) {
meanFieldElements[wIndex * 2 * offset + blockId * size] +=
meanFieldElements[(wIndex * 2 + 1) * offset
+ blockId * size];
wIndex = wIndex + blockDim.x;
}
offset *= 2;
__syncthreads();
}
__syncthreads();

// Mean-field calculation complete - write new spin and delta
if (thrId == 0) {
float meanField = meanFieldElements[blockId * size];
float old = spins[spinId + blockId * size];
if (temp[blockId] > 0) {
spins[spinId + blockId * size] = -1
* tanh(meanField / temp[blockId]) * linearCoef
+ spins[spinId + blockId * size]
* (1 - linearCoef);
} else if (meanField > 0)
spins[spinId + blockId * size] = -1;
else
spins[spinId + blockId * size] = 1;

if (minDiff < fabs(old - spins[spinId + blockId * size]))
continueIteration[blockId] = true; // Too big delta. One more iteration needed
}
__syncthreads();
}
} while (continueIteration[blockId]);
} while (temp[blockId] >= 0);
}