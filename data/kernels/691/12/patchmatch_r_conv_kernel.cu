#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void patchmatch_r_conv_kernel( float *input, float *target, float *conv, int patch, int stride, int c1, int h1, int w1, int h2, int w2 )
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size1 = h1 * w1, size2 = h2 * w2;
int N = size1 * size2;
// id = id1 * size2 + id2

if (id < N) {
int id1 = id / size2, id2 = id % size2;

int x1 = id1 % w1, y1 = id1 / w1;
int x2 = id2 % w2, y2 = id2 / w2;

int kernel_radius = (patch - 1) / 2;

double conv_result = 0, norm_1 = 0, norm_2 = 0;
for (int dy = -kernel_radius; dy <= kernel_radius; dy+=stride) {
for (int dx = -kernel_radius; dx <= kernel_radius; dx+=stride) {
int xx1 = x1 + dx, yy1 = y1 + dy;
int xx2 = x2 + dx, yy2 = y2 + dy;
if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2)
{
int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
for (int c = 0; c < c1; c++) {
float term1 = input[c * size1 + _id1];
float term2 = target[c * size2 + _id2];
conv_result += term1 * term2;
norm_1      += term1 * term1;
norm_2      += term2 * term2;
}

}
}
}

norm_1 = sqrt(norm_1);
norm_2 = sqrt(norm_2);

conv[id] = conv_result / (norm_1 * norm_2 + 1e-9);
}

return ;
}