#include "includes.h"

//#define __OUTPUT_PIX__

#define BLOCK_SIZE 32
__constant__ __device__ float lTable_const[1064];
__constant__ __device__ float mr_const[3];
__constant__ __device__ float mg_const[3];
__constant__ __device__ float mb_const[3];


__global__ void trianguler_convolution_gpu_kernel(float *dev_I, float *dev_O, float *T0, float *T1, float *T2, int wd, int ht, float nrm, float p)
{
unsigned int x_pos = threadIdx.x + (blockDim.x * blockIdx.x);
unsigned int y_pos = threadIdx.y + (blockDim.y * blockIdx.y);

if ((x_pos < wd) && (y_pos < ht)) {

float *It0, *It1, *It2, *Im0, *Im1, *Im2, *Ib0, *Ib1, *Ib2;
float *Ot0, *Ot1, *Ot2;
float *T00, *T10, *T20;


It0 = Im0 = Ib0 = dev_I + (y_pos * wd) + (0 * ht * wd);
It1 = Im1 = Ib1 = dev_I + (y_pos * wd) + (1 * ht * wd);
It2 = Im2 = Ib2 = dev_I + (y_pos * wd) + (2 * ht * wd);

Ot0 = dev_O + (y_pos * wd) + (0 * ht * wd);
Ot1 = dev_O + (y_pos * wd) + (1 * ht * wd);
Ot2 = dev_O + (y_pos * wd) + (2 * ht * wd);

T00 = T0 + (y_pos * wd);
T10 = T1 + (y_pos * wd);
T20 = T2 + (y_pos * wd);

if(y_pos > 0) { /// not the first row, let It point to previous row
It0 -= wd;
It1 -= wd;
It2 -= wd;
}
if(y_pos < ht - 1) { /// not the last row, let Ib point to next row
Ib0 += wd;
Ib1 += wd;
Ib2 += wd;
}

T00[x_pos] = nrm * (It0[x_pos] + (p * Im0[x_pos]) + Ib0[x_pos]);
T10[x_pos] = nrm * (It1[x_pos] + (p * Im1[x_pos]) + Ib1[x_pos]);
T20[x_pos] = nrm * (It2[x_pos] + (p * Im2[x_pos]) + Ib2[x_pos]);

__syncthreads();

if (x_pos == 0) {
Ot0[x_pos] = ((1 + p) * T00[x_pos]) + T00[x_pos + 1];
Ot1[x_pos] = ((1 + p) * T10[x_pos]) + T10[x_pos + 1];
Ot2[x_pos] = ((1 + p) * T20[x_pos]) + T20[x_pos + 1];
} else if (x_pos == wd - 1) {
Ot0[x_pos] = T00[x_pos - 1] + ((1 + p) * T00[x_pos]);
Ot1[x_pos] = T10[x_pos - 1] + ((1 + p) * T10[x_pos]);
Ot2[x_pos] = T20[x_pos - 1] + ((1 + p) * T20[x_pos]);
} else {
Ot0[x_pos] = T00[x_pos - 1] + (p * T00[x_pos]) + T00[x_pos + 1];
Ot1[x_pos] = T10[x_pos - 1] + (p * T10[x_pos]) + T10[x_pos + 1];
Ot2[x_pos] = T20[x_pos - 1] + (p * T20[x_pos]) + T20[x_pos + 1];
}

__syncthreads();
}

}