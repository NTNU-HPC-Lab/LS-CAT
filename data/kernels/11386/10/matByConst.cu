#include "includes.h"
__global__ void matByConst(unsigned char *img, unsigned char *result, int alpha, int cols, int rows) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows && col < cols) {
int idx = row * cols + col;
result[idx] = img[idx] * alpha;
}
}