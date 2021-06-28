#include "includes.h"

#define INF 2147483647

extern "C" {





}
__global__ void oneMove(int * tab, int dist, int pow, int blocksPerTask, int period) {
__shared__ int tmp_T[1024];
__shared__ int begin;

if(threadIdx.x == 0)
begin = (blockIdx.x/blocksPerTask)*dist*2 + (blockIdx.x%blocksPerTask)*512*pow;

__syncthreads();

if((blockIdx.x / period) % 2 == 0) {
for(int i = begin; i < begin + pow*512; i += 512) {
if(threadIdx.x < 512) tmp_T[threadIdx.x] = tab[i + threadIdx.x];
else tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + dist];

__syncthreads();

if(threadIdx.x < 512 && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + 512]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
}

__syncthreads();

if(threadIdx.x < 512) tab[i + threadIdx.x] = tmp_T[threadIdx.x];
else tab[i + threadIdx.x - 512 + dist] = tmp_T[threadIdx.x];

__syncthreads();
}
} else {
for(int i = begin; i < begin + pow*512; i += 512) {
if(threadIdx.x < 512) tmp_T[threadIdx.x] = tab[i + threadIdx.x];
else tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + dist];

__syncthreads();

if(threadIdx.x < 512 && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + 512]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
}

__syncthreads();

if(threadIdx.x < 512) tab[i + threadIdx.x] = tmp_T[threadIdx.x];
else tab[i + threadIdx.x - 512 + dist] = tmp_T[threadIdx.x];

__syncthreads();
}
}
}