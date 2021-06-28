#include "includes.h"
/* Kintsakis Athanasios AEM 6667 */

#define inf 9999




__global__ void funct(int n, int k, float* x, int* qx)
{

int ix= blockIdx.x*blockDim.x + threadIdx.x;

//Epeksigisi
/*
float temp2=x[i*n+k] + x[k*n+j];
omws
i=ix/n;

kai
j=ix%n = ix&(n-1)
i*n = ix/n * n = ix-ix%n= ix-j

*/


int j=ix&(n-1);
float temp2=x[ix-j+k]+x[k*n+j];

if(x[ix]>temp2)
{
x[ix]=temp2;
qx[ix]=k;
}

if(x[ix]==inf)
{
qx[ix]=-2;
}

}