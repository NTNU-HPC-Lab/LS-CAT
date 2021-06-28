#include "includes.h"
//Source: https://kb.iu.edu/d/bdmg
//INDIANA UNIVERSITY
/**********************    mat_mul.cu    ******************************/

#define M  256
#define P  128
#define N   64
#define BLKSIZ 16


/**********************************************************************/
__global__ void mat_mul(float *Ad, float *Bd, float *Cd) {
int    m = blockIdx.x;
int    n = blockIdx.y;
int    i = threadIdx.x;
int    j = threadIdx.y;
int    k,p;
float  c = 0.0;

__shared__  float As[BLKSIZ][BLKSIZ];
__shared__  float Bs[BLKSIZ][BLKSIZ];

for(p=0;p<P/BLKSIZ;p++) {
As[i][j] = Ad[(m*BLKSIZ+i)*P+(p*BLKSIZ+j)];
Bs[i][j] = Bd[(p*BLKSIZ+i)*N+(n*BLKSIZ+j)];
__syncthreads();
for(k=0; k<BLKSIZ; k++) {
c += As[i][k] * Bs[k][j];
}
}
Cd[(m*BLKSIZ+i)*N+(n*BLKSIZ+j)] = c;
}