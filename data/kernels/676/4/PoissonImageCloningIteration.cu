#include "includes.h"
__device__ void assign_add(float *target, const float *source)
{
target[0] += source[0];
target[1] += source[1];
target[2] += source[2];
}
__global__ void	PoissonImageCloningIteration( const float *fixed, const float *mask, const float *source, float *target ,const int wt, const int ht)
{
const int yt = blockIdx.y * blockDim.y + threadIdx.y;
const int xt = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = wt*yt+xt;
const int Nt = wt*(yt-1)+xt;
const int Wt = wt*yt+xt-1;
const int St = wt*(yt+1)+xt;
const int Et = wt*yt+xt+1;
float sum[3] = {};
if(yt < ht and xt < wt){
assign_add(sum, &fixed[curt*3]);
if((yt-1) >= 0){
if(mask[Nt] > 127.0f){
assign_add(sum, &source[Nt*3]);
}
}
if((xt-1) >= 0){
if(mask[Wt] > 127.0f){
assign_add(sum, &source[Wt*3]);
}
}
if((yt+1) < ht){
if(mask[St] > 127.0f){
assign_add(sum, &source[St*3]);
}
}
if((xt+1) < wt){
if(mask[Et] > 127.0f){
assign_add(sum, &source[Et*3]);
}
}
target[curt*3+0] = sum[0]/4;
target[curt*3+1] = sum[1]/4;
target[curt*3+2] = sum[2]/4;
}
}