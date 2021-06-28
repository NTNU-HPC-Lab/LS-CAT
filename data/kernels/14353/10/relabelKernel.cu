#include "includes.h"
__global__ void relabelKernel(int *components, int previousLabel, int newLabel, const int colsComponents) {
uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

if (components[i * colsComponents + j] == previousLabel) {
components[i * colsComponents + j] = newLabel;
}
}