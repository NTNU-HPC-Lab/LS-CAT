#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void patchmatch2_conv_kernel( float *A, float *B, float *AP, float *BP, float *conv, int *prev_corrAB_upsampled, int patch, int s_rad, int c, int h, int w )
{
int h1 = h, h2 = h, w1 = w, w2 = w;
int _id = blockIdx.x * blockDim.x + threadIdx.x;
int size1 = h * w, size2 = h * w;
int s_size = 2 * s_rad + 1;
int s_n = s_size * s_size;
if (_id < size1 * s_n) {
conv[_id] = -1;

int id1 = _id / s_n, s_idx = _id % s_n;
int y1 = id1 / w1, x1 = id1 % w1;
int dy2 = s_idx / s_size - s_rad, dx2 = s_idx % s_size - s_rad;

int x2 = prev_corrAB_upsampled[2 * id1 + 0];
int y2 = prev_corrAB_upsampled[2 * id1 + 1];

int new_y2 = y2 + dy2;
int new_x2 = x2 + dx2;
if (!(new_x2 >= 0 && new_x2 < w2 && new_y2 >= 0 && new_y2 < h2)) {
return ;
}

// Improve by local searching
int kernel_radius = (patch - 1) / 2;
float conv_result = 0;
int cnt = 0;
for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
int xx1 = x1 + dx, yy1 = y1 + dy;
int xx2 = new_x2 + dx, yy2 = new_y2 + dy;
if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2)
{
int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
for (int dc = 0; dc < c; dc++) {
float term1 = A[dc * size1 + _id1];
float term2 = B[dc * size2 + _id2];
conv_result += term1 * term2;

term1 = AP[dc * size1 + _id1];
term2 = BP[dc * size2 + _id2];
conv_result += term1 * term2;
}
cnt++;

}
}
}

conv[_id] = conv_result / cnt;
}
return ;
}