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

__global__ void matrixMul(double * a,double * b, double * C, int cols,int rows,int cols2)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row < rows && col < cols)
{
C[row*cols+col]  =0;
for (int k = 0; k < cols2; k++)
{
C[row*cols+col]+=b[k*cols+col]*a[row*cols2+k];
}
}
}