#include "includes.h"
__global__ void random(unsigned int seed, int* result) {
/* CUDA's random number library uses curandState_t to keep track of the seed value
we will store a random state for every thread  */
curandState_t state;

/* we have to initialize the state */
curand_init(seed, /* the seed controls the sequence of random values that are produced */
0, /* the sequence number is only important with multiple cores */
0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
&state);

/* curand works like rand - except that it takes a state as a parameter */
*result = curand(&state) % MAX;
}