#include "includes.h"
__device__ float Dist_between_two_vec(float * v0, float *v1, int size) {
float dist = 0;
for (int i = 0; i < size; i++)
dist += (v0[i] - v1[i])*(v0[i] - v1[i]);

return sqrt(dist);
}
__global__ void NN_naive(float * A, int colsA, int sizeA, float * B, int colsB, int numsB, int dim, float * idx, float * dist) {
float tmp_dist = 99999;
int nn_id = -1;
int idA = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

for (int idB = 0; idB < (numsB*colsB); idB += colsB) {
float adist = Dist_between_two_vec(A + colsA*idA, B + colsB*idB, dim);
if (tmp_dist > adist) {
tmp_dist = adist;
nn_id = idB;
}
}
*(dist + idA) = tmp_dist;
*(idx + idA) = nn_id;
}