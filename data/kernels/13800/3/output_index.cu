#include "includes.h"



#define ITER 4
#define BANK_OFFSET1(n) (n) + (((n) >> 5))
#define BANK_OFFSET(n) (n) + (((n) >> 5))
#define NUM_BLOCKS(length, dim) nextPow2(length) / (2 * dim)
#define ELEM 4
#define TOTAL_THREADS 512
#define TWO_PWR(n) (1 << (n))
extern float toBW(int bytes, float sec);

__device__ __inline__ void prefix_sum_warp(int thid, int* temp, int N)
{
if (thid < 16)
{
int i = temp[thid];
if (thid >= 1) temp[thid] += temp[thid - 1];
if (thid >= 2) temp[thid] += temp[thid - 2];
if (thid >= 4) temp[thid] += temp[thid - 4];
if (thid >= 8) temp[thid] += temp[thid - 8];
temp[thid] -= i;
}
}
__global__ void output_index(int* device_hist, int* pdevice_hist, int* device_input, int* device_out, int length, int num_blocks, int nibble)
{
__shared__ int temp[TWO_PWR(ITER)];
int t = 4 * blockIdx.x * blockDim.x + threadIdx.x;
int N = TOTAL_THREADS;
int thid = threadIdx.x;

if (t < length)
{
int val1;
int val2;
int val3;
int val4;
int nibble1 = nibble << 2;
int lindex1;
int lindex2;
int lindex3;
int lindex4;
int gindex1;
int gindex2;
int gindex3;
int gindex4;
int a = t;
int b = t + 1 * N;
int c = t + 2 * N;
int d = t + 3 * N;
int a1 = thid;
int b1 = thid + 1 * N;
int c1 = thid + 2 * N;
int d1 = thid + 3 * N;
val1 = device_input[a];
val2 = device_input[b];
val3 = device_input[c];
val4 = device_input[d];

if (thid < 32)
{
if ((thid) < ITER)
{
temp[4 * thid] = device_hist[4 * thid * num_blocks + blockIdx.x];
temp[4 * thid + 1] = device_hist[(4 * thid + 1) * num_blocks + blockIdx.x];
temp[4 * thid + 2] = device_hist[(4 * thid + 2) * num_blocks + blockIdx.x];
temp[4 * thid + 3] = device_hist[(4 * thid + 3) * num_blocks + blockIdx.x];
}

prefix_sum_warp(thid, temp, TWO_PWR(ITER));
}
__syncthreads();
lindex1 = temp[((val1 >> (nibble1)) & ((1 << ITER) - 1))];
lindex2 = temp[((val2 >> (nibble1)) & ((1 << ITER) - 1))];
lindex3 = temp[((val3 >> (nibble1)) & ((1 << ITER) - 1))];
lindex4 = temp[((val4 >> (nibble1)) & ((1 << ITER) - 1))];
gindex1 = pdevice_hist[((val1 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
gindex2 = pdevice_hist[((val2 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
gindex3 = pdevice_hist[((val3 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];
gindex4 = pdevice_hist[((val4 >> (nibble1)) & ((1 << ITER) - 1)) * num_blocks + blockIdx.x];

device_out[a1 + gindex1 - lindex1] = val1;
device_out[b1 + gindex2 - lindex2] = val2;
device_out[c1 + gindex3 - lindex3] = val3;
device_out[d1 + gindex4 - lindex4] = val4;
}
}