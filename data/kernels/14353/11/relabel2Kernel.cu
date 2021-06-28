#include "includes.h"
__global__ void relabel2Kernel(int *components, int previousLabel, int newLabel, const int colsComponents, const int idx, const int frameRows) {
uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
i = i * colsComponents + j;
i = i + (colsComponents * frameRows * idx);
if (components[i] == previousLabel) {
components[i] = newLabel;
}

}