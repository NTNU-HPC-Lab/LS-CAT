#include "includes.h"
__global__ void matrixSubScalarInverse(double *a, double b, double *c, int cr, int cc){

int x = blockIdx.x * blockDim.x + threadIdx.x; // col
int y = blockIdx.y * blockDim.y + threadIdx.y; // row


if(x < cc && y < cr){

c[y * cc + x] = b-a[y * cc + x];
}

}