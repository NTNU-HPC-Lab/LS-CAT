#include "includes.h"
__global__ void relabelUnrollKernel(int *components, int previousLabel, int newLabel, const int colsComponents, const int idx, const int frameRows, const int factor) {
uint id_i_child = (blockIdx.x * blockDim.x) + threadIdx.x;
id_i_child = id_i_child + (frameRows * idx);
uint id_j_child = (blockIdx.y * blockDim.y) + threadIdx.y;
id_j_child = (colsComponents / factor) * id_j_child;
uint i = id_i_child;
for (int j = id_j_child; j < (colsComponents / factor); j++) {
if (components[i * colsComponents + j] == previousLabel) {
components[i * colsComponents + j] = newLabel;
}
}
}