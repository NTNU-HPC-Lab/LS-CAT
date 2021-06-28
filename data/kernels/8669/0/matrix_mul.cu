#include "includes.h"
/*
* Alexandre Maros - 2016
*
* Cuda Matrix Multiplication with Global Memory.
*
* nvcc cuda_matrix_global.cu -o cg.o
*
* Implemented by Alexandre Maros for learning purposes.
* A version of this code using Shared Memory is in here:
* https://github.com/alepmaros/cuda_matrix_multiplication
*
* Distributed under the MIT Lincese.
*/


//32x32
#define NTHREADS_X 16
#define NTHREADS_Y 32
#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

/* A macro used for error checking in CUDA function calls
* Credit to: http://stackoverflow.com/a/14038590 for the gpuErrchk macro.
*/
__global__ void matrix_mul(int *a, int *b, int *c, int a_ncolumns, int c_nlines, int c_ncolumns)
{

int column = blockIdx.x * blockDim.x + threadIdx.x;
int line =  blockIdx.y * blockDim.y + threadIdx.y;

if (column  >= c_ncolumns || line >= c_nlines)
return;

int i, sum = 0;


int beginA = a_ncolumns * line;
int beginB = column;

for (i = 0; i < a_ncolumns; i++)
{
sum += a[beginA + i] * b[i * c_ncolumns + beginB];
}

c[line * c_ncolumns + column] = sum;
}