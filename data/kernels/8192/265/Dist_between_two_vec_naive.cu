#include "includes.h"
__device__ float Dist_between_two_vec(float * v0, float *v1, int size) {
float dist = 0;
for (int i = 0; i < size; i++)
dist += (v0[i] - v1[i])*(v0[i] - v1[i]);

return sqrt(dist);
}
__global__ void Dist_between_two_vec_naive(float * v0, float *v1, int size, float * dst) {
float dist = 0;
for (int i = 0; i < size; i++)
dist += (v0[i] - v1[i]);//*(v0[i]-v1[i]);

dst[0] = dist;
}