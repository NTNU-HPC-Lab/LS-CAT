#include "includes.h"
__global__ void computeMinEnergyMatrix(float *energy, float *min_energy, int height, int width) {
const int bid = blockIdx.x;
const int tid = threadIdx.x;
const int pos = bid * blockDim.x + tid;

extern __shared__ float shared_row_energy[];

if(pos < width) {
for(int i=tid; i<width; i += blockDim.x) {
shared_row_energy[i] = energy[i];
min_energy[i] = energy[i];
}
} else {
return;
}
__syncthreads();

float temp[4];
for(int i=1; i<height; i++) {
int k = 0;
for(int j=tid; j<width; j += blockDim.x) {
float l = (j==0) ? 999999999 :  shared_row_energy[j-1];
float m = shared_row_energy[j];
float r = (j==width-1) ? 999999999: shared_row_energy[j+1];

float minimum = energy[i*width + j] + min(l, min(m, r));
temp[k++] = minimum;
}
__syncthreads();
k = 0;
for(int j=tid; j<width; j += blockDim.x) {
shared_row_energy[j] = temp[k];
min_energy[i * width + j] = temp[k++];
}
}
}