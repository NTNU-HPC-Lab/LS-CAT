#include "includes.h"
__global__ void cosineKernel(float *a, float *b, float *outN, float *outD1, float *outD2, int size) {
extern __shared__ float sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
int stride = gridDim.x * blockDim.x;
while (i < size) {
sdata[3 * tid] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
sdata[3 * tid + 1] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
sdata[3 * tid + 2] = a[i] * b[i] + a[i + blockDim.x] * b[i + blockDim.x];
__syncthreads();
for (unsigned int s = blockDim.x / 2; s > 96; s >>= 1) {
if (tid < s) {
sdata[3 * tid] += sdata[3 * tid + s];
sdata[3 * tid + 1] += sdata[3 * tid + s + 1];
sdata[3 * tid + 2] += sdata[3 * tid + s + 2];
}
}
if (tid < 32) {
sdata[3 * tid] += sdata[3 * tid + 96];
sdata[3 * tid + 1] += sdata[3 * tid + 97];
sdata[3 * tid + 2] += sdata[3 * tid + 98];
sdata[3 * tid] += sdata[3 * tid + 48];
sdata[3 * tid + 1] += sdata[3 * tid + 49];
sdata[3 * tid + 2] += sdata[3 * tid + 50];
sdata[3 * tid] += sdata[3 * tid + 24];
sdata[3 * tid + 1] += sdata[3 * tid + 25];
sdata[3 * tid + 2] += sdata[3 * tid + 26];
sdata[3 * tid] += sdata[3 * tid + 12];
sdata[3 * tid + 1] += sdata[3 * tid + 13];
sdata[3 * tid + 2] += sdata[3 * tid + 14];
sdata[3 * tid] += sdata[3 * tid + 6];
sdata[3 * tid + 1] += sdata[3 * tid + 7];
sdata[3 * tid + 2] += sdata[3 * tid + 8];
sdata[3 * tid] += sdata[3 * tid + 3];
sdata[3 * tid + 1] += sdata[3 * tid + 4];
sdata[3 * tid + 2] += sdata[3 * tid + 5];
}
if (tid == 0) {
outN[blockIdx.x] = sdata[0];
outD1[blockIdx.x] = sdata[1];
outD2[blockIdx.x] = sdata[2];
}
i += stride;
}
//if (blockSize >= 512) {
//	if (tid < 256) {
//		sndata[tid] += sndata[tid + 256];
//		sd1data[tid] += sd1data[tid + 256];
//		sd2data[tid] += sd2data[tid + 256];
//	} __syncthreads();
//}
//if (blockSize >= 256) {
//	if (tid < 128) {
//		sndata[tid] += sndata[tid + 128];
//		sd1data[tid] += sd1data[tid + 128];
//		sd2data[tid] += sd2data[tid + 128];
//	} __syncthreads();
//}
//if (blockSize >= 128) {
//	if (tid < 64) {
//		sndata[tid] += sndata[tid + 64];
//		sd1data[tid] += sd1data[tid + 64];
//		sd2data[tid] += sd2data[tid + 64];
//	} __syncthreads();
//}
//if (tid < 32) {
//	if (blockSize >= 64) {
//		sndata[tid] += sndata[tid + 32];
//		sd1data[tid] += sd1data[tid + 32];
//		sd2data[tid] += sd2data[tid + 32];
//	}
//	if (blockSize >= 32) {
//		sndata[tid] += sndata[tid + 16];
//		sd1data[tid] += sd1data[tid + 16];
//		sd2data[tid] += sd2data[tid + 16];
//	}
//	if (blockSize >= 16) {
//		sndata[tid] += sndata[tid + 8];
//		sd1data[tid] += sd1data[tid + 8];
//		sd2data[tid] += sd2data[tid + 8];
//	}
//	if (blockSize >= 8) {
//		sndata[tid] += sndata[tid + 4];
//		sd1data[tid] += sd1data[tid + 4];
//		sd2data[tid] += sd2data[tid + 4];
//	}
//	if (blockSize >= 4) {
//		sndata[tid] += sndata[tid + 2];
//		sd1data[tid] += sd1data[tid + 2];
//		sd2data[tid] += sd2data[tid + 2];
//	}
//	if (blockSize >= 2) {
//		sndata[tid] += sndata[tid + 1];
//		sd1data[tid] += sd1data[tid + 1];
//		sd2data[tid] += sd2data[tid + 1];
//	}
//}
}