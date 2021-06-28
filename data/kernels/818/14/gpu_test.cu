#include "includes.h"
__global__ void gpu_test(unsigned char* Pout, unsigned char* Pin, int width, int height) {
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
int i = row * width + col;

if (row < height && col < width) {
Pout[i] = Pin[i];
}

}