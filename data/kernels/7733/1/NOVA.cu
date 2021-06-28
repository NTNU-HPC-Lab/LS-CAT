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

__global__ void NOVA(double * Beta,double * Inverse,int * Vec, int p0,double Sigma2)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
double t0,Pvalue;
t0=Beta[x]/sqrt(Sigma2*Inverse[x*p0+x]);
Pvalue=2.*(1.-erf(t0));
if(Pvalue<0.25)
{
Vec[x]=1;
}
else
{
Vec[x]=0;
}
}