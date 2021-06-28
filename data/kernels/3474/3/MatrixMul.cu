#include "includes.h"
__global__ void MatrixMul(float *darray_1, float *darray_2 , float *dres_arr, int n){
// cols and rows definition
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
// Mat mult operation
for(int i = 0; i<n; i++){
dres_arr[row*n+col]+= darray_1[row*n+i]*darray_2[col+i*n];
// printf("row %i * height %i col %i index %i res %f\n", row, n, col, i, dres_arr[row*n+col]);
}
}