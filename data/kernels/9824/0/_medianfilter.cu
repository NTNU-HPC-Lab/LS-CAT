#include "includes.h"

#define N 33 * 1024
#define threadsPerBlock 256
#define blocksPerGrid (N + threadsPerBlock - 1) / threadsPerBlock
#define RADIUS 2
// Signal/image element type
typedef int element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     N      - length of the signal



//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     N      - length of the signal
__global__ void _medianfilter(const element* signal, element* result)
{
__shared__ element cache[threadsPerBlock + 2 * RADIUS];
element window[5];
int gindex = threadIdx.x + blockDim.x * blockIdx.x;
int lindex = threadIdx.x + RADIUS;
// Reads input elements into shared memory
cache[lindex] = signal[gindex];
if (threadIdx.x < RADIUS)
{
cache[lindex - RADIUS] = signal[gindex - RADIUS];
cache[lindex + threadsPerBlock] = signal[gindex + threadsPerBlock];
}
__syncthreads();
for (int j = 0; j < 2 * RADIUS + 1; ++j)
window[j] = cache[threadIdx.x + j];
// Orders elements (only half of them)
for (int j = 0; j < RADIUS + 1; ++j)
{
// Finds position of minimum element
int min = j;
for (int k = j + 1; k < 2 * RADIUS + 1; ++k)
if (window[k] < window[min])
min = k;
// Puts found minimum element in its place
const element temp = window[j];
window[j] = window[min];
window[min] = temp;
}
// Gets result - the middle element
result[gindex] = window[RADIUS];
}