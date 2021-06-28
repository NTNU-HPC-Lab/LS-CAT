#include "includes.h"
/* Kintsakis Athanasios AEM 6667 */

#define inf 9999




__global__ void funct2(int n, int k, float* x, int* qx)
{

int ix= blockIdx.x*blockDim.x + threadIdx.x;

int j=ix&(n-1);
float temp2=x[ix-j+k]+x[k*n+j];
if(x[ix]>temp2)
{
x[ix]=temp2;
qx[ix]=k;
}

}