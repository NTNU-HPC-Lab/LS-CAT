#include "includes.h"
__device__ static float flow_absolute_residual(float x, float y, float ux, float uy, float d, float fx, float fy, float T0, float T1, float T2, float R0, float R1, float R2) {
float rx = -ux + fx * R1 - y * R2 + ((x * x) * R1) / fx + d * fx * T0 -
d * x * T2 - (x * y * R0) / fx;
float ry = -uy - fy * R0 + x * R2 - d * y * T2 - ((y * y) * R0) / fy +
d * fy * T1 + (x * y * R1) / fy;

return sqrtf(rx * rx + ry * ry);
}
__global__ void flow_absolute_residual_scalable_GPU( float *d_abs_res, const float2 *d_flow_compact, const float *d_Zbuffer_flow_compact, const int *d_ind_flow_Zbuffer, const unsigned int *d_valid_flow_Zbuffer, float fx, float fy, float ox, float oy, int n_rows, int n_cols, int n_valid_flow_Zbuffer, const int *d_offset_ind, const int *d_segment_translation_table, float w_flow, float w_ar_flow, const float *d_dTR) {

int ind = blockDim.x * blockIdx.x + threadIdx.x;

if (ind < n_valid_flow_Zbuffer) {

// determine current segment
int segment = d_segment_translation_table[d_valid_flow_Zbuffer[ind]];

// fetch flow and Zbuffer from global memory
float2 u = d_flow_compact[ind];
float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[ind]);

// compute coordinates
int pixel_ind = d_ind_flow_Zbuffer[ind];
bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

float y = floorf(__fdividef((float)pixel_ind, n_cols));
float x = (float)pixel_ind - y * n_cols;

x = x - ox;
y = y - oy;

// compute absolute residual
// here the weights will be introduced
float w = is_ar_flow ? w_ar_flow : w_flow;
int ind_out = ind + d_offset_ind[segment];
int s6 = segment * 6;
d_abs_res[ind_out] =
w * flow_absolute_residual(x, y, u.x, u.y, disp, fx, fy, d_dTR[s6],
d_dTR[s6 + 1], d_dTR[s6 + 2], d_dTR[s6 + 3],
d_dTR[s6 + 4], d_dTR[s6 + 5]);
}
}