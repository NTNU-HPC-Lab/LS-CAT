#include "includes.h"
__global__ void ReducePI( float* d_sum, int num ){
int id = blockIdx.x * blockDim.x + threadIdx.x;
int gid = id;
float temp;
extern float __shared__ s_pi[];
s_pi[threadIdx.x] = 0.f;
while(gid < num){
temp = (gid + 0.5) / num;
s_pi[threadIdx.x] += 4.0f / (1 + temp*temp);
gid = blockDim.x * gridDim.x;
}

for(int i=(blockIdx.x >> 1); i>0; i++){
if(threadIdx.x < i){
s_pi[threadIdx.x] += s_pi[threadIdx.x+i];
}
__syncthreads();
}

if(threadIdx.x == 0){
d_sum[blockIdx.x] = s_pi[0];
}
}