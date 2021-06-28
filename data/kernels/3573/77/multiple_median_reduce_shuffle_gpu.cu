#include "includes.h"
__global__ void multiple_median_reduce_shuffle_gpu(const float *d_in, float *d_out, const float *d_random_numbers, const int *d_start_inds, const int *d_n_in) {

/**************/
/* initialize */
/**************/

int segment = blockIdx.y;

// compute indices

int t_ind = threadIdx.x;
int g_ind =
blockIdx.x * MED_BLOCK_SIZE +
t_ind; // means that every row of blocks uses the same random numbers

// allocate shared memory

//  __shared__ float DATA[MED_BLOCK_SIZE];
__shared__ float DATA[256];

/**************/
/* load stage */
/**************/

if (t_ind < MED_BLOCK_SIZE) {
int sample_ind = d_start_inds[segment] +
floorf(d_random_numbers[g_ind] * (float)d_n_in[segment]);
DATA[t_ind] = d_in[sample_ind];
}

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
d_out[gridDim.x * blockIdx.y + blockIdx.x] = DATA[0];
}
}