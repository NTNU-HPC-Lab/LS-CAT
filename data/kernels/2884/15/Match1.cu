#include "includes.h"
__global__ void Match1(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
int p1 = threadIdx.x + M1W*blockIdx.x;
float max_score = 0.0f;
int index = -1;

for (int p2=0;p2<NPTS;p2++) {
float score = 0.0f;
for (int d=0;d<NDIM;d++)
score += d_pts1[p1*NDIM + d]*d_pts2[p2*NDIM + d];
if (score>max_score) {
max_score = score;
index = p2;
}
}

d_score[p1] = max_score;
d_index[p1] = index;
}