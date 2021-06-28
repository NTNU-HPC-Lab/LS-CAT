#include "includes.h"


__global__ void polynomial_expansion (float* poly,int degree,int n,float* array)
{
int INX=blockIdx.x*blockDim.x+threadIdx.x;
if(INX<n)
{
float val=0.0;
float exp=1.0;
for(int x=0;x<=degree;++x)
{
val+=exp*poly[x];
exp*=array[INX];
}
array[INX]=val;
}
}