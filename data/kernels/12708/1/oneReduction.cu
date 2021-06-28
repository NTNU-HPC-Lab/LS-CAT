#include "includes.h"

#define INF 2147483647

extern "C" {





}
__global__ void oneReduction(int * tab, int len, int mod) {

__shared__ int begin, end;
__shared__ int tmp_T[1024];

if(threadIdx.x == 0) {
begin = blockIdx.x*len;
end = blockIdx.x*len + len;
}

__syncthreads();

if(blockIdx.x % mod < mod/2) {
for(int k = len/2; k >= 1024; k /= 2) {
for(int g = begin; g < end; g += 2*k) {
for(int j = g; j < g + k; j += 512) {
__syncthreads();

if(threadIdx.x < 512)
tmp_T[threadIdx.x] = tab[j + threadIdx.x];
else
tmp_T[threadIdx.x] = tab[j + threadIdx.x - 512 + k];

__syncthreads();
if(threadIdx.x < 512 && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + 512]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
}

__syncthreads();
if(threadIdx.x < 512)
tab[j + threadIdx.x] = tmp_T[threadIdx.x];
else
tab[j + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];
}
}
}

for(int i = begin; i < begin+len; i += 1024) {
__syncthreads();
tmp_T[threadIdx.x] = tab[i + threadIdx.x];
__syncthreads();
for(int jump = 512; jump >= 1; jump /= 2) {
if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + jump]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
}
__syncthreads();
}
tab[i + threadIdx.x] = tmp_T[threadIdx.x];
}
} else {
for(int k = len/2; k >= 1024; k /= 2) {
for(int g = begin; g < end; g += 2*k) {
for(int j = g; j < g + k; j += 512) {
__syncthreads();
if(threadIdx.x < 512)
tmp_T[threadIdx.x] = tab[j + threadIdx.x];
else
tmp_T[threadIdx.x] = tab[j + threadIdx.x - 512 + k];

__syncthreads();
if(threadIdx.x < 512 && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + 512]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
}

__syncthreads();
if(threadIdx.x < 512)
tab[j + threadIdx.x] = tmp_T[threadIdx.x];
else
tab[j + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];
}
}
}

for(int i = begin; i < begin + len; i += 1024) {
__syncthreads();
tmp_T[threadIdx.x] = tab[i + threadIdx.x];
__syncthreads();
for(int jump = 512; jump >= 1; jump /= 2) {
if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + jump]) {
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
}
__syncthreads();
}
tab[i + threadIdx.x] = tmp_T[threadIdx.x];
}
}


}