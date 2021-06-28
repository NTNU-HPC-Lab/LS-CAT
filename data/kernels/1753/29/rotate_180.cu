#include "includes.h"
__global__ void rotate_180( float* data,int nx, int nxy, int offset, unsigned int size) {

const uint x=threadIdx.x;
const uint y=blockIdx.x;

__shared__ float shared_lower_data[MAX_THREADS];
__shared__ float shared_upper_data[MAX_THREADS];

shared_lower_data[x] = data[x+y*MAX_THREADS+offset];
shared_upper_data[x] = data[nxy + x+(-y-1)*MAX_THREADS-offset];
__syncthreads();


if (size == 0) {
float tmp = shared_lower_data[x];
shared_lower_data[x] = shared_upper_data[MAX_THREADS-x-1];
shared_upper_data[MAX_THREADS-x-1] = tmp;
} else {
if ( x < size ) {
float tmp = shared_lower_data[x];
shared_lower_data[x] = shared_upper_data[MAX_THREADS-x-1];
shared_upper_data[MAX_THREADS-x-1]= tmp;

}
}

__syncthreads();
if (size == 0) {
data[x+y*MAX_THREADS+offset] = shared_lower_data[x];
data[nxy+x+(-y-1)*MAX_THREADS-offset] = shared_upper_data[x];
} else {
if ( x < size ) {
data[nxy-x-1+(-y)*MAX_THREADS-offset] = shared_upper_data[MAX_THREADS-x-1];
data[x+y*MAX_THREADS+offset] = shared_lower_data[x];
}
}

}