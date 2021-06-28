#include "includes.h"
#define NUM_THREADS 32














__global__ void binary_kernel_same(const float * vg_a, size_t pitch_a, size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b, size_t k, float * d, size_t pitch_d, float p)
{
size_t x = blockIdx.x;
size_t y = blockIdx.y;

if(x == y && x < n_a && threadIdx.x == 0) {
d[y * pitch_d + x] = 0.0;
}

// If all element is to be computed
if(y < n_a && x < y) {
__shared__ float temp[2 * NUM_THREADS];

temp[threadIdx.x] = 0.0;
temp[threadIdx.x + NUM_THREADS] = 0.0;
for(size_t offset = threadIdx.x; offset < k; offset += blockDim.x) {
int a = vg_a[x * pitch_a + offset] != 0.0;
int b = vg_a[y * pitch_a + offset] != 0.0;
if(a ^ b) {
temp[threadIdx.x] += 1.0;
}
if(a || b) {
temp[threadIdx.x + NUM_THREADS] += 1.0;
}
}

// Sync with other threads
__syncthreads();

// Reduce
for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
if(threadIdx.x < stride) {
temp[threadIdx.x] += temp[threadIdx.x + stride];
temp[threadIdx.x + NUM_THREADS] += temp[threadIdx.x + stride + NUM_THREADS];
}
__syncthreads();
}
// Write to global memory
if(threadIdx.x == 0) {
float val = temp[0];
if(temp[NUM_THREADS] != 0.0) {
val /= temp[NUM_THREADS];
}
d[y * pitch_d + x] = val;
d[x * pitch_d + y] = val;
}
}
}