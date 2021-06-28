#include "includes.h"
#define rows 1000
#define cols 1000

// CUDA kernel. Each thread takes care of one element of c

__global__ void matricesMul(double *m1, double *m2, double *m3)
{
// Get our global thread ID
int ti = blockIdx.y*blockDim.y+threadIdx.y;
int tj = blockIdx.x*blockDim.x+threadIdx.x;
// Make sure we do not go out of bounds
if(ti < rows && tj < cols){
double data= 0.0;
for(int k=0;k<rows;k++) data += m1[ti*rows+k] * m2[k*cols+tj];
m3[ti*rows+tj] = data;
}
}