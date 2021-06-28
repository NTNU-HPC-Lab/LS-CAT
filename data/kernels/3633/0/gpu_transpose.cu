#include "includes.h"
int row = 0;
int col = 0;
using namespace std;

__global__
__global__ void gpu_transpose(float *dst, float *A, int col, int row) {
int idx = threadIdx.x + blockIdx.x*blockDim.x;

if(idx<col){
for (int j=0; j<row; j++){
dst[j*col+idx] = A[idx*row+j];
}
}
}