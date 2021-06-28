#include "includes.h"
__global__ void compute_distance_texture(cudaTextureObject_t ref, int                 ref_width, float *             query, int                 query_width, int                 query_pitch, int                 height, float*              dist) {
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if ( xIndex<query_width && yIndex<ref_width) {
float ssd = 0.f;
for (int i=0; i<height; i++) {
float tmp  = tex2D<float>(ref, (float)yIndex, (float)i) - query[i * query_pitch + xIndex];
ssd += tmp * tmp;
}
dist[yIndex * query_pitch + xIndex] = ssd;
}
}