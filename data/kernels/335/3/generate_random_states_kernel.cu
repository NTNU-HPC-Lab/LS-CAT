#include "includes.h"
__global__ void generate_random_states_kernel(unsigned int seed, curandState_t* d_states, size_t total_number) {
int idx = threadIdx.x + blockIdx.x * blockDim.x;
// int idx_g = idx;
if (idx < total_number) {
curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
idx, /* the sequence number should be different for each core (unless you want all
cores to get the same sequence of numbers for some reason - use thread id! */
0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
&d_states[idx]);

__syncthreads();
}
}