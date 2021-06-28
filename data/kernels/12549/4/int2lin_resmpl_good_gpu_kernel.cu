#include "includes.h"

//#define __OUTPUT_PIX__

#define BLOCK_SIZE 32
__constant__ __device__ float lTable_const[1064];
__constant__ __device__ float mr_const[3];
__constant__ __device__ float mg_const[3];
__constant__ __device__ float mb_const[3];


__global__ void int2lin_resmpl_good_gpu_kernel(float *dev_in_img, float *dev_out_img, float *dev_C0_tmp, float *dev_C1_tmp, float *dev_C2_tmp, int org_wd, int org_ht, int dst_wd, int dst_ht, int n_channels, float r, int *yas_const, int *ybs_const)
{

unsigned int x_pos = threadIdx.x + (blockDim.x * blockIdx.x);
unsigned int y_pos = threadIdx.y + (blockDim.y * blockIdx.y);

if ((x_pos < dst_wd) && (y_pos < dst_ht)) {

int ya, yb;
float *A00, *A01, *A02, *A03, *B00;
float *A10, *A11, *A12, *A13, *B10;
float *A20, *A21, *A22, *A23, *B20;

float *A0 = dev_in_img + 0;
float *B0 = dev_out_img + (0 * dst_ht * dst_wd);
float *A1 = dev_in_img + 1;
float *B1 = dev_out_img + (1 * dst_ht * dst_wd);
float *A2 = dev_in_img + 2;
float *B2 = dev_out_img + (2 * dst_ht * dst_wd);

if (org_ht == dst_ht && org_wd == dst_wd) {
int out_img_idx = y_pos + (dst_wd * x_pos);
B0[out_img_idx] = A0[out_img_idx * n_channels];
B1[out_img_idx] = A1[out_img_idx * n_channels];
B2[out_img_idx] = A2[out_img_idx * n_channels];
return;
}

int y1 = 0;

if (org_ht == 2 * dst_ht) {
y1 += 2 * y_pos;
} else if (org_ht == 3 * dst_ht) {
y1 += 3 * y_pos;
} else if (org_ht == 4 * dst_ht) {
y1 += 4 * y_pos;
}

if (y_pos == 0)
y1 = 0;

ya = yas_const[y1];
A00 = A0 + (ya * org_wd * n_channels);
A01 = A00 + (org_wd * n_channels);
A02 = A01 + (org_wd * n_channels);
A03 = A02 + (org_wd * n_channels);

A10 = A1 + (ya * org_wd * n_channels);
A11 = A00 + (org_wd * n_channels);
A12 = A01 + (org_wd * n_channels);
A13 = A02 + (org_wd * n_channels);

A20 = A2 + (ya * org_wd * n_channels);
A21 = A00 + (org_wd * n_channels);
A22 = A01 + (org_wd * n_channels);
A23 = A02 + (org_wd * n_channels);

yb = ybs_const[y1];
B00 = B0 + (yb * dst_wd);
B10 = B1 + (yb * dst_wd);
B20 = B2 + (yb * dst_wd);

// resample along y direction
if (org_ht == 2 * dst_ht) {
dev_C0_tmp[x_pos] = A00[x_pos * n_channels] + A01[x_pos * n_channels];
dev_C1_tmp[x_pos] = A10[x_pos * n_channels] + A11[x_pos * n_channels];
dev_C2_tmp[x_pos] = A20[x_pos * n_channels] + A21[x_pos * n_channels];
} else if (org_ht == 3 * dst_ht) {
dev_C0_tmp[x_pos] = A00[x_pos * n_channels] + A01[x_pos * n_channels] + A02[x_pos * n_channels];
dev_C1_tmp[x_pos] = A10[x_pos * n_channels] + A11[x_pos * n_channels] + A12[x_pos * n_channels];
dev_C2_tmp[x_pos] = A20[x_pos * n_channels] + A21[x_pos * n_channels] + A22[x_pos * n_channels];
} else if (org_ht == 4 * dst_ht) {
dev_C0_tmp[x_pos] = A00[x_pos * n_channels] + A01[x_pos * n_channels] + A02[x_pos * n_channels] + A03[x_pos * n_channels];
dev_C1_tmp[x_pos] = A10[x_pos * n_channels] + A11[x_pos * n_channels] + A12[x_pos * n_channels] + A13[x_pos * n_channels];
dev_C2_tmp[x_pos] = A20[x_pos * n_channels] + A21[x_pos * n_channels] + A22[x_pos * n_channels] + A23[x_pos * n_channels];
}

/* ensure that all threads have calculated the values for C until this point */
__syncthreads();

// resample along x direction (B -> C)
if (org_wd == 2 * dst_wd) {
B00[x_pos]= (dev_C0_tmp[2 * x_pos] + dev_C0_tmp[(2 * x_pos) + 1]) * (r / 2);
B10[x_pos]= (dev_C1_tmp[2 * x_pos] + dev_C1_tmp[(2 * x_pos) + 1]) * (r / 2);
B20[x_pos]= (dev_C2_tmp[2 * x_pos] + dev_C2_tmp[(2 * x_pos) + 1]) * (r / 2);
} else if (org_wd == 3 * dst_wd) {
B00[x_pos] = (dev_C0_tmp[3 * x_pos] + dev_C0_tmp[(3 * x_pos) + 1] + dev_C0_tmp[(3 * x_pos) + 2]) * (r / 3);
B10[x_pos] = (dev_C1_tmp[3 * x_pos] + dev_C1_tmp[(3 * x_pos) + 1] + dev_C1_tmp[(3 * x_pos) + 2]) * (r / 3);
B20[x_pos] = (dev_C2_tmp[3 * x_pos] + dev_C2_tmp[(3 * x_pos) + 1] + dev_C2_tmp[(3 * x_pos) + 2]) * (r / 3);
} else if (org_wd == 4 * dst_wd) {
B00[x_pos] = (dev_C0_tmp[4 * x_pos] + dev_C0_tmp[(4 * x_pos) + 1] + dev_C0_tmp[(4 * x_pos) + 2] + dev_C0_tmp[(4 * x_pos) + 3]) * (r / 4);
B10[x_pos] = (dev_C1_tmp[4 * x_pos] + dev_C1_tmp[(4 * x_pos) + 1] + dev_C1_tmp[(4 * x_pos) + 2] + dev_C1_tmp[(4 * x_pos) + 3]) * (r / 4);
B20[x_pos] = (dev_C2_tmp[4 * x_pos] + dev_C2_tmp[(4 * x_pos) + 1] + dev_C2_tmp[(4 * x_pos) + 2] + dev_C2_tmp[(4 * x_pos) + 3]) * (r / 4);
}

__syncthreads();
}
}