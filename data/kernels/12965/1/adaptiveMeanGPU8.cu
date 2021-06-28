#include "includes.h"
__global__ void adaptiveMeanGPU8 (float* D, int32_t D_width, int32_t D_height) {

// Global coordinates and Pixel id
uint32_t u0 = blockDim.x*blockIdx.x + threadIdx.x + 4;
uint32_t v0 = blockDim.y*blockIdx.y + threadIdx.y + 4;
uint32_t idx = v0*D_width + u0;
//Local thread coordinates
uint32_t ut = threadIdx.x + 4;
uint32_t vt = threadIdx.y + 4;

//If out of filter range return instantly
if(u0 > (D_width - 4) || v0 > (D_height - 4))
return;

//Allocate Shared memory array with an appropiate margin for the bitlateral filter
//Since we are using 8 pixels with the center pixel being 5,
//we need 4 extra on left and top and 3 extra on right and bottom
__shared__ float D_shared[32+7][32+7];
//Populate shared memory
if(threadIdx.x == blockDim.x-1){
D_shared[ut+1][vt] = D[idx+1];
D_shared[ut+2][vt] = D[idx+2];
D_shared[ut+3][vt] = D[idx+3];
//D_shared[ut+4][vt] = D[idx+4];
}
if(threadIdx.x == 0){
D_shared[ut-4][vt] = D[idx-4];
D_shared[ut-3][vt] = D[idx-3];
D_shared[ut-2][vt] = D[idx-2];
D_shared[ut-1][vt] = D[idx-1];
}
if(threadIdx.y == 0){
D_shared[ut][vt-4] = D[(v0-4)*D_width+u0];
D_shared[ut][vt-3] = D[(v0-3)*D_width+u0];
D_shared[ut][vt-2] = D[(v0-2)*D_width+u0];
D_shared[ut][vt-1] = D[(v0-1)*D_width+u0];
}
if(threadIdx.y == blockDim.y-1){
D_shared[ut][vt+1] = D[(v0+1)*D_width+u0];
D_shared[ut][vt+2] = D[(v0+2)*D_width+u0];
D_shared[ut][vt+3] = D[(v0+3)*D_width+u0];
//D_shared[ut][vt+4] = D[(v0+4)*D_width+u0];
}

if(D[idx] < 0){
// zero input disparity maps to -10 (this makes the bilateral
// weights of all valid disparities to 0 in this region)
D_shared[ut][vt] = -10;
}else{
D_shared[ut][vt] = D[idx];
}
__syncthreads();

// full resolution: 8 pixel bilateral filter width
// D(x) = sum(I(xi)*f(I(xi)-I(x))*g(xi-x))/W(x)
// W(x) = sum(f(I(xi)-I(x))*g(xi-x))
// g(xi-x) = 1
// f(I(xi)-I(x)) = 4-|I(xi)-I(x)| if greater than 0, 0 otherwise
// horizontal filter

// Current pixel being filtered is middle of our set (4 back, in orginal its 3 for some reason)
//Note this isn't truely the center since original uses 8 vectore resisters
float val_curr = D_shared[ut][vt];

float weight_sum0 = 0;
float weight_sum = 0;
float factor_sum = 0;

for(int32_t i=0; i < 8; i++){
weight_sum0 = 4.0f - fabs(D_shared[ut+(i-4)][vt]-val_curr);
weight_sum0 = max(0.0f, weight_sum0);
weight_sum += weight_sum0;
factor_sum += D_shared[ut+(i-4)][vt]*weight_sum0;
}

if (weight_sum>0) {
float d = factor_sum/weight_sum;
if (d>=0) *(D+idx) = d;
}

__syncthreads();
//Update shared memory
if(threadIdx.x == blockDim.x-1){
D_shared[ut+1][vt] = D[idx+1];
D_shared[ut+2][vt] = D[idx+2];
D_shared[ut+3][vt] = D[idx+3];
//D_shared[ut+4][vt] = D[idx+4];
}
if(threadIdx.x == 0){
D_shared[ut-4][vt] = D[idx-4];
D_shared[ut-3][vt] = D[idx-3];
D_shared[ut-2][vt] = D[idx-2];
D_shared[ut-1][vt] = D[idx-1];
}
if(threadIdx.y == 0){
D_shared[ut][vt-4] = D[(v0-4)*D_width+u0];
D_shared[ut][vt-3] = D[(v0-3)*D_width+u0];
D_shared[ut][vt-2] = D[(v0-2)*D_width+u0];
D_shared[ut][vt-1] = D[(v0-1)*D_width+u0];
}
if(threadIdx.y == blockDim.y-1){
D_shared[ut][vt+1] = D[(v0+1)*D_width+u0];
D_shared[ut][vt+2] = D[(v0+2)*D_width+u0];
D_shared[ut][vt+3] = D[(v0+3)*D_width+u0];
//D_shared[ut][vt+4] = D[(v0+4)*D_width+u0];
}

if(D[idx] < 0){
D_shared[ut][vt] = -10;
}else{
D_shared[ut][vt] = D[idx];
}

__syncthreads();

// vertical filter
// set pixel of interest
val_curr = D_shared[ut][vt];

weight_sum0 = 0;
weight_sum = 0;
factor_sum = 0;

for(int32_t i=0; i < 8; i++){
weight_sum0 = 4.0f - fabs(D_shared[ut][vt+(i-4)]-val_curr);
weight_sum0 = max(0.0f, weight_sum0);
weight_sum += weight_sum0;
factor_sum += D_shared[ut][vt+(i-4)]*weight_sum0;
}

if (weight_sum>0) {
float d = factor_sum/weight_sum;
if (d>=0) *(D+idx) = d;
}

}