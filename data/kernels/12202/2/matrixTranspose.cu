#include "includes.h"
__global__ void matrixTranspose(unsigned int* A_d, unsigned int *T_d, int n) {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// **** Populate matrixTranspose kernel function ****
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
int i = threadIdx.x;
int j = threadIdx.y;

if(i<n&&j<n)
T_d[i+j*n] = A_d[j+i*n];

}