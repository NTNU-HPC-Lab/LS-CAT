#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void patchmatch_r_argmax_kernel( float *conv, float *target, float *match, int *correspondence, int c1, int h1, int w1, int h2, int w2 )
{
int id1 = blockIdx.x * blockDim.x + threadIdx.x;
int size1 = h1 * w1, size2 = h2 * w2;

if (id1 < size1) {
//int x1 = id1 % w1, y1 = id1 / w1;
double conv_max = -1e20;

for (int y2 = 0; y2 < h2; y2++) {
for (int x2 = 0; x2 < w2; x2++) {
int id2 = y2 * w2 + x2;

int id = id1 * size2 + id2;
float conv_result = conv[id];

if (conv_result > conv_max) {
conv_max = conv_result;
correspondence[id1 * 2 + 0] = x2;
correspondence[id1 * 2 + 1] = y2;
for (int c = 0; c < c1; c++) {
match[c * size1 + id1] = target[c * size2 + id2];
}
}
}
}

}

return ;
}