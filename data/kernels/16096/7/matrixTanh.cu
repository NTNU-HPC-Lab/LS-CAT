#include "includes.h"
__global__ void matrixTanh(double *a, double *c, int cr, int cc){

int x = blockIdx.x * blockDim.x + threadIdx.x; // col
int y = blockIdx.y * blockDim.y + threadIdx.y; // row

if(x < cc && y < cr){
c[y * cc + x] = tanh(a[y * cc + x]);
}

}