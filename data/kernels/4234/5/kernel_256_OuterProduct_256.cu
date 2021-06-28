#include "includes.h"
__global__ void kernel_256_OuterProduct_256(float *A, float *B, float *C) {
int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
int c_input = tY*256 + tX, c_kernel = c_input, T_offset = (Tile<<12) + (Part<<11) + c_input, B_offset = (Tile<<16) + c_kernel;

extern __shared__ float input[];
float *kernel = input + 2048, *out = kernel + 8192;
int B_stride[32] = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936};
out[c_input] = 0.0f;
out[c_input+1024] = 0;

input[c_input] = A[T_offset];
input[c_input+1024] = A[T_offset+1024];

for (int k = 0; k < 8; k++) {
int B_start = B_offset + (k<<13); // 32*64
kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
kernel[c_kernel+4096] = B[B_start+4096], kernel[c_kernel+5120] = B[B_start+5120];
kernel[c_kernel+6144] = B[B_start+6144], kernel[c_kernel+7168] = B[B_start+7168];

__syncthreads();

float sum = 0, sum1 = 0;
int y_tmp = (tY<<8)+(k<<5), y_tmp1 = y_tmp+1024;
for (int j = 0; j < 32; j++) {
sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
sum1 += input[y_tmp1 + j] * kernel[tX + B_stride[j]];
}
out[c_input] += sum;
out[c_input+1024] += sum1;
__syncthreads();
}

C[T_offset] = out[c_input];
C[T_offset+1024] = out[c_input+1024];
}