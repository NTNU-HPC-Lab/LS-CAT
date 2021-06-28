#include "includes.h"
__global__ void Convolution(double* A, double* B, int I, int J)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
double c11, c12, c13, c21, c22, c23, c31, c32, c33;

c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
c13 = +0.4;  c23 = +0.7;  c33 = +0.1;

if (i>J && i<I*J-J && (i%J!=0) && ((i+1)%J!=0)) {
B[i] = c11 * A[i-J-1]  +  c12 * A[i-1]  +  c13 * A[i+J-1]
+ c21 * A[i-J]  +  c22 * A[i]  +  c23 * A[i+J]
+ c31 * A[i-J+1]  +  c32 * A[i+1]  +  c33 * A[i+J+1];
}

}