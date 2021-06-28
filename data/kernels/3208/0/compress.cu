#include "includes.h"
/*
*
*    Carlos Roman Rivera - A01700820
*
*    Programming Languages - Cuda Quiz
*
*/


#define N 9
#define K N/3
#define ThreadsPerBlock K
#define NumBlocks K


__global__ void compress(float *mat, int n, float *comp, int k){
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;

if (row < k && col < k) {
comp[col + row * k] = 0;
for (int i_row = 0 ; i_row < k ; i_row++) {
for (int j_col = 0 ; j_col < k ; j_col++) {
comp[col + row * k] += mat[(col + j_col) + (row + i_row) * n];
}
}
}

}