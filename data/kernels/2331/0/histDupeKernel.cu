#include "includes.h"
__global__ void histDupeKernel(const float* data1, const float* data2, const float* confidence1, const float* confidence2, int* ids1, int* ids2, int* results_id1, int* results_id2, float* results_similarity, int* result_count, const int N1, const int N2, const int max_results) {

const unsigned int thread = threadIdx.x; // Thread index within block
const unsigned int block = blockIdx.x; // Block index
const unsigned int block_size = blockDim.x; // Size of each block

const unsigned int block_start = block_size * block; // Index of the start of the block
const unsigned int index = block_start + thread; // Index of this thread

//__shared__ float conf[64]; // Shared array of confidence values for all histograms owned by this block
//conf[thread] = confidence1[index]; // Coalesced read of confidence values
float conf = confidence1[index];
int id = ids1[index];

__shared__ float hists[128 * 64]; // Shared array of all histograms owned by this block
for (unsigned int i = 0; i < 64; i++) {
hists[i * 128 + thread] = data1[(block_start + i) * 128 + thread]; // Coalesced read of first half of histogram
hists[i * 128 + thread + 64] = data1[(block_start + i) * 128 + 64 + thread]; // Coalesced read of second half of histogram
}

__shared__ float other[128]; // Histogram to compare all owned histograms against parallely
for (unsigned int i = 0; i < N2 && *result_count < max_results; i++) {

other[thread] = data2[i * 128 + thread]; // Coalesced read of first half of other histogram
other[thread + 64] = data2[i * 128 + thread + 64]; // Second half

__syncthreads(); // Ensure all values read

if (index < N1) {
float d = 0;
for (unsigned int k = 0; k < 128; k++) { // Compute sum of distances between thread-owned histogram and shared histogram
d += fabsf(hists[thread * 128 + k] - other[k]);
}
d = 1 - (d / 8); // Massage the difference into a nice % similarity number, between 0 and 1

int other_id = ids2[i];

if (other_id != id && d > fmaxf(conf, confidence2[i])) { // Don't compare against self, only compare using highest confidence
int result_index = atomicAdd(result_count, 1); // Increment result count by one atomically (returns value before increment)
if (result_index < max_results) {
// Store resulting pair
results_similarity[result_index] = d;
results_id1[result_index] = id;
results_id2[result_index] = other_id;
}
}
}

__syncthreads(); // Ensure all threads have finished before looping and reading new shared histogram
}

}