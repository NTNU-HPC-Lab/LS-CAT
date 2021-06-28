#include "includes.h"
__global__ void  blelloch_no_padding(unsigned int* d_in_array, const size_t numBins)
/*

\Params:
* d_in_array - input array of histogram values in each bin. Gets converted
to cdf by the end of the function.
* numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
*/
{

int thid = threadIdx.x;

extern __shared__ float temp_array[];

temp_array[thid] = d_in_array[thid];
temp_array[thid + numBins/2] = d_in_array[thid + numBins/2];

__syncthreads();

// Part 1: Up Sweep, reduction
int stride = 1;
for (int d = numBins>>1; d > 0; d>>=1) {

if (thid < d) {
int neighbor = stride*(2*thid+1) - 1;
int index = stride*(2*thid+2) - 1;

temp_array[index] += temp_array[neighbor];
}
stride *=2;
__syncthreads();
}
// Now set last element to identity:
if (thid == 0)  temp_array[numBins-1] = 0;

// Part 2: Down sweep
for (int d=1; d<numBins; d *= 2) {
stride >>= 1;
__syncthreads();

if(thid < d) {
int neighbor = stride*(2*thid+1) - 1;
int index = stride*(2*thid+2) - 1;

float t = temp_array[neighbor];
temp_array[neighbor] = temp_array[index];
temp_array[index] += t;
}
}

__syncthreads();

d_in_array[thid] = temp_array[thid];
d_in_array[thid + numBins/2] = temp_array[thid + numBins/2];

}