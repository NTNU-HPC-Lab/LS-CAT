#include "includes.h"
__global__ void __full(int *ir, int *ic, double *data, double *od, int nrows, int ncols, int nnz) {
int i, row, col;
double v;
int id = threadIdx.x + blockIdx.x * blockDim.x;
for (i = id; i < nnz; i += blockDim.x * gridDim.x) {
v = data[i];
row = ir[i];
col = ic[i];
od[row + col * nrows] = v;
}
}