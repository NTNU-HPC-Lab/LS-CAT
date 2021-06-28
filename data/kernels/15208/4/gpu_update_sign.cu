#include "includes.h"
__global__ void gpu_update_sign(int *G, double *w ,int *neighbors , int k , int n ,int *temp, int *flag,int it_b ,int it_t)
{
int result;
double sum = 0.0;
int buf=0;
//Find the indexes
int x = blockIdx.x+it_b*gridDim.x;
int y = threadIdx.x+it_t*blockDim.x;

if (blockIdx.x+it_b*gridDim.x<n && threadIdx.x+it_t*blockDim.x<n)
{
//Calculate result
for (int i = 0; i < k; i++){
for (int j = 0; j < k; j++){
sum += ((double)G[neighbors[x*n*k*k+y*k*k+i*k+j]])*w[i*k+j];
}
}
//Evaluate and write back
if ( sum > 1e-6){
result = 1;
if (result != G[neighbors[x*n*k*k+y*k*k+12]])
buf++;
}
else if( sum < -(1e-6)){
result = -1;
if (result != G[neighbors[x*n*k*k+y*k*k+12]])
buf++;
}
else
result = G[neighbors[x*n*k*k+y*k*k+12]];

*flag+=buf;
temp[x*n+y] =result;
}
}