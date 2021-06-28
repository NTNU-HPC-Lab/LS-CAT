#include "includes.h"
__global__ void matrixMultiply(double *a, double *b, double *c, int cr, int cc, int ac, int bc){

long x = blockIdx.x * blockDim.x + threadIdx.x; // col
long y = blockIdx.y * blockDim.y + threadIdx.y; // row
double sum = 0;

if(x < cc && y < cr){

for(int k = 0; k<ac; k++){
sum+= a[y*ac+k] * b[k*bc+x];
}
c[y * cc + x] = sum;
}

}