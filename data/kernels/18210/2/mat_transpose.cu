#include "includes.h"
__global__ void mat_transpose(const float *a, float *b, int n, int m){
const int TIlE_WIDTH = 8;
__shared__ float temp[TIlE_WIDTH][TIlE_WIDTH];

int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;

int i = TIlE_WIDTH * bx + tx;
int j = TIlE_WIDTH * by + ty;
int idxa = j * n + i;
int idxb = i * n + j;

temp[ty][tx] = a[idxa];
__syncthreads();

b[idxb] = temp[ty][tx];

// if(i < n and j < m){
//     b[idxb] = a[idxa];
// }
}