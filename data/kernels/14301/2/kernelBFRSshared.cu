#include "includes.h"
__device__ float maxMetricPoints(const float* g_uquery, const float* g_vpoint, int pointdim, int signallength){
float	r_u1;
float	r_v1;
float	r_d1,r_dim=0;

r_dim=0;
for(int d=0; d<pointdim; d++){
r_u1 = *(g_uquery+d*signallength);
r_v1 = *(g_vpoint+d*signallength);
r_d1 = r_v1 - r_u1;
r_d1 = r_d1 < 0? -r_d1: r_d1;  //abs
r_dim= r_dim < r_d1? r_d1: r_dim;
}
return r_dim;
}
__global__ void kernelBFRSshared(const float* g_uquery, const float* g_vpointset, int *g_npoints, int pointdim, int triallength, int signallength, int exclude, float radius)
{

// shared memory
extern __shared__ char array[];
int *s_npointsrange;
s_npointsrange = (int*)array;

const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
const unsigned int itrial = tid / triallength;  //  indextrial

if(tid<signallength){

s_npointsrange[threadIdx.x] = 0;
__syncthreads();


unsigned int indexi = tid-triallength*itrial;
for(int t=0; t<triallength; t++){
int indexu = tid;
int indexv = (t + itrial*triallength);
int condition1=indexi-exclude;
int condition2=indexi+exclude;
if((t<condition1)||(t>condition2)){
float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
if(temp_dist <= radius){
s_npointsrange[threadIdx.x]++;
}
}

}

__syncthreads();
//printf("\ntid:%d npoints: %d\n",tid, s_npointsrange[threadIdx.x]);
//COPY TO GLOBAL MEMORY
g_npoints[tid] = s_npointsrange[threadIdx.x];

}
}