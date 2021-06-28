#include "includes.h"
/*
:copyright:
William B. Frank and Eric Beauce
:license:
GNU General Public License, Version 3
(https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

__global__ void sum_cc(float *cc_mat, float *cc_sum, float *weights, int n_stations, int n_components, int n_corr, int chunk_offset, int chunk_size) {

int i, ch;

i = blockIdx.x * blockDim.x + threadIdx.x;
if ( ((i + chunk_offset) < n_corr) & (i < chunk_size) ){
// first condition: check if we are not outside cc_sum's length
// second condition: check if we are not outside the chunk's size
float *cc_mat_offset;

cc_mat_offset = cc_mat + i * n_stations * n_components;
for (ch = 0; ch < (n_stations * n_components); ch++) cc_sum[i] += cc_mat_offset[ch] * weights[ch];
}
}