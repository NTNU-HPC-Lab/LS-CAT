#include "includes.h"
//Library Definition

//Constant Definition
#define PI 3.141592654
#define blocksize 32
#define Repetitions 8192


//Print matrix into standard output
void print(double * M,int cols,int rows);
void dot(double * a,double * b, double & c, int cols);
void Create_New_Matrix(double * M,double * New,int * vec, int p0, int pp,int nn);

/*
DEVICE FUNCTIONS
*/

//Matrix transposition (Rows and Cols of M)

__global__ void gaussjordan(double *A, double *I, int nn, int i)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if ( x< nn && y < nn)
{
if (x < nn && y < nn)
{
if (x != i)
{
I[x*nn + y] -= I[i*nn + y] * A[x*nn + i];
if (y != i)
{
A[x*nn + y] -= A[i*nn + y] * A[x*nn + i];
}
}
}
}
}