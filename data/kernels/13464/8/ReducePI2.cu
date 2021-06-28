#include "includes.h"
__global__ void ReducePI2( float* d_sum, int num, float* d_pi ){
int id=threadIdx.x;
extern float __shared__ s_sum[];
s_sum[id]=d_sum[id];
__syncthreads();
for(int i=(blockDim.x>>1);i>0;i>>=1){
if(id<i)
s_sum[id]+=s_sum[id+i];
__syncthreads();
}
printf("%d,%f\n",id,s_sum[id]);
if(id==0){
*d_pi=s_sum[0]/num;
printf("%d,%f\n",id,*d_pi);
}

}