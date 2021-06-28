#include "includes.h"
__global__ void initCURAND(unsigned int seed, curandState_t* d_states) {

/* we have to initialize the state */
curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
blockIdx.x, /* the sequence number should be different for each core (unless you want all
cores to get the same sequence of numbers for some reason - use thread id! */
0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
&d_states[blockIdx.x]);
}