#include "includes.h"

//#define __OUTPUT_PIX__

#define BLOCK_SIZE 32
__constant__ __device__ float lTable_const[1064];
__constant__ __device__ float mr_const[3];
__constant__ __device__ float mg_const[3];
__constant__ __device__ float mb_const[3];


__global__ void convert_to_luv_gpu_kernel(unsigned char *in_img, float *out_img, int cols, int rows, bool use_rgb)
{
float r, g, b, l, u, v, x, y, z, lt;

unsigned int x_pos = threadIdx.x + (blockDim.x * blockIdx.x);
unsigned int y_pos = threadIdx.y + (blockDim.y * blockIdx.y);

if ((x_pos < cols) && (y_pos < rows)) {

unsigned int pos = (y_pos * cols) + x_pos;

if (use_rgb) {
r = (float)in_img[(3 * pos)];
g = (float)in_img[(3 * pos) + 1];
b = (float)in_img[(3 * pos) + 2];
} else {
b = (float)in_img[(3 * pos)];
g = (float)in_img[(3 * pos) + 1];
r = (float)in_img[(3 * pos) + 2];
}

x = (mr_const[0] * r) + (mg_const[0] * g) + (mb_const[0] * b);
y = (mr_const[1] * r) + (mg_const[1] * g) + (mb_const[1] * b);
z = (mr_const[2] * r) + (mg_const[2] * g) + (mb_const[2] * b);

float maxi = 1.0f / 270;
float minu = -88.0f * maxi;
float minv = -134.0f * maxi;
float un = 0.197833f;
float vn = 0.468331f;

lt = lTable_const[static_cast<int>((y*1024))];
l = lt; z = 1/(x + (15 * y) + (3 * z) + (float)1e-35);
u = lt * (13 * 4 * x * z - 13 * un) - minu;
v = lt * (13 * 9 * y * z - 13 * vn) - minv;

out_img[(3 * pos)] = l;
out_img[(3 * pos) + 1] = u;
out_img[(3 * pos) + 2] = v;
}
}