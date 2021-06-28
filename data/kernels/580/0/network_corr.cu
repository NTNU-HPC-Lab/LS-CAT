#include "includes.h"
/*
:copyright:
William B. Frank and Eric Beauce
:license:
GNU General Public License, Version 3
(https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

__global__ void network_corr(float *templates, float *sum_square_template, int *moveout, float *data, float *weights, size_t step, size_t n_samples_template, size_t n_samples_data, size_t n_stations, size_t n_components, int chunk_offset, int chunk_size, float *cc_mat) {

// each thread matches the template to one time in the data
int idx, first_sample_block, first_sample_trace, last_sample_trace; // sample's index
int i, s, c; // counters
int data_offset, templates_offset, sum_square_template_offset, cc_mat_offset;
float numerator, denominator, sum_square_data;
float data_sample;
int t_idx;

//------------------------------------------------
int count_template = (n_samples_template / WARPSIZE + 1) * WARPSIZE;
extern __shared__ float shared[];
float *ss_template = &shared[0];
float *templates_s = &shared[sizeof(float)];
float *data_s = &shared[count_template+sizeof(float)];

// 1 block processes one channel to blockDim.x / step different positions in time
idx = blockIdx.x/n_stations * blockDim.x + chunk_offset;
first_sample_block = idx * step;
s = blockIdx.x % n_stations;

for (c = 0; c < n_components; c++){
if (weights[s * n_components + c] != 0.){
// compute offsets for input variables
cc_mat_offset = (first_sample_block / step + threadIdx.x - chunk_offset) * n_stations * n_components + s * n_components + c;
templates_offset = s * n_samples_template * n_components + c * n_samples_template;
sum_square_template_offset = s * n_components + c;
first_sample_trace = first_sample_block + moveout[s * n_components + c];
last_sample_trace = first_sample_trace + n_samples_template + threadIdx.x * step;
data_offset = s * n_samples_data * n_components + c * n_samples_data + first_sample_trace;

// initialize sums
sum_square_data = 0.0f;
numerator = 0.0f;

// load template and data into shared memory
t_idx = threadIdx.x;
if (t_idx == 0){
ss_template[0] = sum_square_template[sum_square_template_offset];
}
while(t_idx < n_samples_template) {
templates_s[t_idx] = templates[templates_offset + t_idx];
if ((first_sample_trace + t_idx) < n_samples_data) data_s[t_idx] = data[data_offset + t_idx];
t_idx += blockDim.x;
}
while(t_idx < (blockDim.x * step + n_samples_template)){
if ((first_sample_trace + t_idx) < n_samples_data) data_s[t_idx] = data[data_offset + t_idx];
t_idx += blockDim.x;
}

__syncthreads(); // make sure the waveforms are read before keep going

// calculate correlation coefficient
if (last_sample_trace < n_samples_data){
// if not, corresponds to an ill-defined CC with some samples out of the bounds
for(i = 0; i < n_samples_template; i++) {
data_sample = data_s[i + threadIdx.x * step];
numerator += data_sample * templates_s[i];
sum_square_data += data_sample * data_sample;
}
//denominator = sum_square_data * sum_square_template[sum_square_template_offset];
denominator = sum_square_data * ss_template[0];
if (cc_mat_offset < (chunk_size * n_stations * n_components)){
// check that this thread is not ouf of the chunk's bounds
if (denominator > STABILITY_THRESHOLD) cc_mat[cc_mat_offset] = numerator * rsqrtf(denominator);
}
}
}
__syncthreads(); // wait for every thread to finish before leaving the kernel
}
}