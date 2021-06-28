#include "includes.h"
__global__ void findMax(int *m, int *cs, int n)
{
// your code goes here
int colnum = blockDim.x * blockIdx.x + threadIdx.x;
int max = m[0];
for (int    k = 0; k < n; k++){
if(m [colnum+n*k] > max)
max = m [colnum+n*k];
}
cs[colnum] = max;
}