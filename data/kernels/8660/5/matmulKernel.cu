#include "includes.h"
__global__ void matmulKernel(float* mat1,float* mat2, float* matP,int dim)	{
int thread_x,thread_y,i;
thread_x=blockIdx.x*blockDim.x+threadIdx.x;
thread_y=blockIdx.y*blockDim.y+threadIdx.y;
if(thread_x<dim&&thread_y<dim)	{
float P_value=0.;
for(i=0;i<dim;i++)	{
P_value+=mat1[thread_y*dim+i]*mat2[i*dim+thread_x];
}
matP[thread_y*dim+thread_x]=P_value;
}
}