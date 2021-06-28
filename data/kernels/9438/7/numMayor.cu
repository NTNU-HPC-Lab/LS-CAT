#include "includes.h"
__global__ void numMayor(float *d_v, float *d_pos){


float temp = 0,pos=0;
for(int i=threadIdx.x; i<blockDim.x;i++){
if(d_v[i] > temp){
temp = d_v[i];
pos = i;
}

}
__syncthreads();
if(pos>d_pos[threadIdx.x])
d_pos[threadIdx.x] = pos;
d_v[threadIdx.x] = temp;

}