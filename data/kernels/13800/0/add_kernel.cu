#include "includes.h"



#define ITER 4
#define BANK_OFFSET1(n) (n) + (((n) >> 5))
#define BANK_OFFSET(n) (n) + (((n) >> 5))
#define NUM_BLOCKS(length, dim) nextPow2(length) / (2 * dim)
#define ELEM 4
#define TOTAL_THREADS 512
#define TWO_PWR(n) (1 << (n))
extern float toBW(int bytes, float sec);

__global__ void add_kernel(int* device_result, int* device_blocksum)
{
int temp1;
int thid = threadIdx.x;
int N = blockDim.x;
int offset = blockIdx.x * 4 * blockDim.x;

temp1 = device_blocksum[blockIdx.x];
device_result[offset + thid] = device_result[offset + thid] + temp1;
device_result[offset + thid + N] = device_result[offset + thid + N] + temp1;
device_result[offset + thid + 2 * N] = device_result[offset + thid + 2 * N] + temp1;
device_result[offset + thid + 3 * N] = device_result[offset + thid + 3 * N] + temp1;
}