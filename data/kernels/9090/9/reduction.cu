#include "includes.h"
__global__ void reduction(const int N, float *a, float *result) {

int thread = threadIdx.x;
int block  = blockIdx.x;
int blockSize = blockDim.x;
int gridSize = gridDim.x;


//unique global thread ID
int id = thread + block*blockSize;

__volatile__ __shared__ float s_sum[256];

float sum = 0;
for (int i=0; i<4; i++){
if(id+i*blockSize*gridSize<N){
sum += a[id+i*blockSize*gridSize]; //add the thread's id to start
}
}
s_sum[thread] = sum;

__syncthreads(); //make sure the write to shared is finished

if (thread<128) {//first half
s_sum[thread] += s_sum[thread+128];
}

__syncthreads(); //make sure the write to shared is finished


if (thread<64) {//next half
s_sum[thread] += s_sum[thread+64];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<32) {//next half
s_sum[thread] += s_sum[thread+32];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<16) {//next half
s_sum[thread] += s_sum[thread+16];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<8) {//next half
s_sum[thread] += s_sum[thread+8];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<4) {//next half
s_sum[thread] += s_sum[thread+4];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<2) {//next half
s_sum[thread] += s_sum[thread+2];
}

__syncthreads(); //make sure the write to shared is finished

if (thread<1) {//final piece
s_sum[thread] += s_sum[thread+1];
result[block] = s_sum[thread];
}
}