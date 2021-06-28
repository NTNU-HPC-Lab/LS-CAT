#include "includes.h"
__global__ void median_reduce_shuffle_gpu(const float *d_in, float *d_out, float *d_random_numbers, int n_in) {

/**************/
/* initialize */
/**************/

// compute indices

int t_ind = threadIdx.x;
int g_ind = blockIdx.x * MED_BLOCK_SIZE + t_ind;

// allocate shared memory

__shared__ float DATA[MED_BLOCK_SIZE];

/**************/
/* load stage */
/**************/

int sample_ind = floorf(d_random_numbers[g_ind] * (float)n_in);
DATA[t_ind] = d_in[sample_ind];

__syncthreads();

/*******************/
/* reduction stage */
/*******************/

for (int s = 1; s < MED_BLOCK_SIZE; s *= 3) {

int index = 3 * s * t_ind;

if (index < MED_BLOCK_SIZE) {

// fetch three values
float value1 = DATA[index];
float value2 = DATA[index + s];
float value3 = DATA[index + 2 * s];

// extract the middle value (median)
float smallest = fminf(value1, value2);
value2 = fmaxf(value1, value2);
value1 = smallest;

value3 = fmaxf(value1, value3);
value2 = fminf(value2, value3);

DATA[index] = value2;
}

__syncthreads();
}

/***************/
/* write stage */
/***************/

// write this block's approx median (first element)

if (t_ind == 0) {
d_out[blockIdx.x] = DATA[0];
}
}