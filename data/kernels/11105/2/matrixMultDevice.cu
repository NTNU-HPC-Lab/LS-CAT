#include "includes.h"
__global__ void matrixMultDevice(float* d_A, float* d_B, float* d_C, int width) {
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;
if(Row < width && Col < width) {
float ans = 0.0;
for(int k=0; k<width; k++) {
ans += d_A[Row*width+k]*d_B[k*width+Col];
}
d_C[Row*width+Col]=ans;
}
}