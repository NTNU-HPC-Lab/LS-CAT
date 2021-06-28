#include "includes.h"
__device__ unsigned int cuda_delta = 0;  __device__ unsigned int maskForMode(unsigned int x, unsigned int y, unsigned int z, unsigned int w ){
unsigned int max = x > y ? x : y;
max = z > max ? z : max;
max = w > max ? w : max;
unsigned int mask = 0;

if (max == x){
mask |= 1;
}
if (max == y){
mask |= 2; // 010
}
if (max == z){
mask |= 4; // 0100
}
if (max == w){
mask |= 8; // 0100
}
return mask;
}
__global__ void update_centroids(const sequence_t *data, sequence_t *centroids, unsigned int * tmp_centroidCount, unsigned int numClusters){
int i = blockIdx.x * blockDim.x  + threadIdx.x;

if (i < numClusters){
sequence_t seq = make_ulong3(0,0,0);
unsigned int *tmp_centroid = &tmp_centroidCount[i* BIT_SIZE_OF(sequence_t)];
for (int j=0;j<SEQ_DIM_BITS_SIZE;j+=4)
{

// bits tmp_centroid[0] is less significative bit from sequence_t
// bits tmp_centroid[0] = z << 0
unsigned int *bitCountX = &tmp_centroid[j + (SEQ_DIM_BITS_SIZE * 2)];
unsigned int *bitCountY = &tmp_centroid[j + SEQ_DIM_BITS_SIZE];
unsigned int *bitCountZ = &tmp_centroid[j];

unsigned long int mask = maskForMode(bitCountX[0],bitCountX[1],bitCountX[2],bitCountX[3]);
seq.x |= (mask << j);
mask = maskForMode(bitCountY[0],bitCountY[1],bitCountY[2],bitCountY[3]);
seq.y |= (mask << j);
mask = maskForMode(bitCountZ[0],bitCountZ[1],bitCountZ[2],bitCountZ[3]);
seq.z |= (mask << j);
}
centroids[i] = seq;
}
}