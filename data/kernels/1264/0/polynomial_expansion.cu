#include "includes.h"


__global__ void polynomial_expansion (float* poly,int degree,int n,float* array)
{
int idx=blockIdx.x*blockDim.x+threadIdx.x;
if(idx<n)
{
float val=0.0;
float exp=1.0;
for(int x=0;x<=degree;++x)
{
val+=exp*poly[x];
exp*=array[idx];
}
array[idx]=val;
}
}