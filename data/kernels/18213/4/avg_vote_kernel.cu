#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void avg_vote_kernel( float *A, float *B, int *corrAB, int patch, int c, int h, int w )
{
int _id = blockIdx.x * blockDim.x + threadIdx.x;
int size = h * w;
int radius = patch / 2;
if (_id < c * size) {
// _id = dc * size + id
int id = _id % size, dc = _id / size;
int x1 = id % w, y1 = id / w;
double sum = 0;
int    cnt = 0;
for (int dx = -radius; dx <= radius; dx++) {
for (int dy = -radius; dy <= radius; dy++) {
int new_x1 = x1 + dx, new_y1 = y1 + dy;

if (new_x1 >= 0 && new_x1 < w && new_y1 >= 0 && new_y1 < h) {
int new_id1 = new_y1 * w + new_x1;
int x2 = corrAB[new_id1 * 2 + 0];
int y2 = corrAB[new_id1 * 2 + 1];
int new_x2 = x2 - dx, new_y2 = y2 - dy;

if (new_x2 >= 0 && new_x2 < w && new_y2 >= 0 && new_y2 < h) {
int new_id2 = new_y2 * w + new_x2;
sum += A[dc * size + new_id2];
cnt++;
}
}
}
}
if (cnt != 0)
B[dc * size + id] = sum / cnt;

}
return ;
}