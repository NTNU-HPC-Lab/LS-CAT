#include "includes.h"
__global__ void Product (float *a, float *b, float *c)
{
// Out of all the threads created each one computes 1 value of C and stores into cval

float cval = 0.00;
int R = blockIdx.y * blockDim.y + threadIdx.y; //Row of the matrix
int C = blockIdx.x * blockDim.x + threadIdx.x; //Column of the matrix
//Defining the size of the matrix//
int N=1000;
if(R> N || C > N ){
return;
}
for (int j = 0; j < N; j++)
{
cval += a[R * N+ j] *b[j * N + C];

}
c[R * N + C]+= cval;
}