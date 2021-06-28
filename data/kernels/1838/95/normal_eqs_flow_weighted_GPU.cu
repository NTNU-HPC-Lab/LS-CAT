#include "includes.h"
__device__ static float flow_absolute_residual(float x, float y, float ux, float uy, float d, float fx, float fy, float T0, float T1, float T2, float R0, float R1, float R2) {
float rx = -ux + fx * R1 - y * R2 + ((x * x) * R1) / fx + d * fx * T0 -
d * x * T2 - (x * y * R0) / fx;
float ry = -uy - fy * R0 + x * R2 - d * y * T2 - ((y * y) * R0) / fy +
d * fy * T1 + (x * y * R1) / fy;

return sqrtf(rx * rx + ry * ry);
}
__global__ void normal_eqs_flow_weighted_GPU( float *d_CO, const float2 *d_flow_compact, const float *d_Zbuffer_flow_compact, const int *d_ind_flow_Zbuffer, float fx, float fy, float ox, float oy, int n_rows, int n_cols, const int *d_n_values_flow, const int *d_start_ind_flow, const float *d_abs_res_scales, float w_flow, float w_ar_flow, const float *d_dTR) {

int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
// multiple of blocksize

int n_flow = d_n_values_flow[blockIdx.y];
int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
int start_ind = d_start_ind_flow[blockIdx.y];

// initialize accumulators

float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f;

for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

if (in_ind < n_flow) { // is this a valid sample?

// fetch flow and Zbuffer from global memory
float2 u = d_flow_compact[in_ind + start_ind];
float disp = __fdividef(1.0f, d_Zbuffer_flow_compact[in_ind + start_ind]);

// compute coordinates
int pixel_ind = d_ind_flow_Zbuffer[in_ind + start_ind];
bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

float y = floorf(__fdividef((float)pixel_ind, n_cols));
float x = (float)pixel_ind - y * n_cols;

x = x - ox;
y = y - oy;

// determine M-estimation weight
float w_rel = is_ar_flow ? w_ar_flow : w_flow;
int s6 = blockIdx.y * 6;
float w = w_rel * flow_absolute_residual(x, y, u.x, u.y, disp, fx, fy,
d_dTR[s6], d_dTR[s6 + 1],
d_dTR[s6 + 2], d_dTR[s6 + 3],
d_dTR[s6 + 4], d_dTR[s6 + 5]);
w /= d_abs_res_scales[blockIdx.y];
w = (w > 1) ? 0 : (1.0f - 2.0f * w * w + w * w * w * w);

/************************/
/* evaluate constraints */
/************************/

// unique values A-matrix

A0 += w * (disp * disp * fx * fx);
A1 += w * (-disp * disp * x * fx);
A2 += w * (-disp * x * y);
A3 += w * (disp * fx * fx + disp * x * x);
A4 += w * (-disp * y * fx);
A5 += w * (-disp * disp * y * fy);
A6 += w * (-disp * fy * fy - disp * y * y); //!!!!
A7 += w * (disp * x * fy);
A8 += w * (disp * disp * x * x + disp * disp * y * y);
A9 += w * (disp * x * x * y / fx + disp * y * fy + disp * y * y * y / fy);
A10 +=
w * (-disp * x * fx - disp * x * x * x / fx - disp * x * y * y / fy);
A11 += w * (x * x * y * y / (fx * fx) + fy * fy + 2.0f * y * y +
y * y * y * y / (fy * fy));
A12 += w * (-2.0f * x * y - x * x * x * y / (fx * fx) -
x * y * y * y / (fy * fy));
A13 += w * (x * y * y / fx - x * fy - x * y * y / fy);
A14 += w * (fx * fx + 2.0f * x * x + x * x * x * x / (fx * fx) +
x * x * y * y / (fy * fy));
A15 += w * (-y * fx - x * x * y / fx + x * x * y / fy);
A16 += w * (x * x + y * y);

// B-vector

A17 += w * (disp * u.x * fx);
A18 += w * (disp * u.y * fy);
A19 += w * (-disp * x * u.x - disp * y * u.y);
A20 += w * (-x * y * u.x / fx - u.y * fy - u.y * y * y / fy);
A21 += w * (u.x * fx + x * x * u.x / fx + x * y * u.y / fy);
A22 += w * (-y * u.x + x * u.y);
}
}

/**************************/
/* write out accumulators */
/**************************/

int out_ind =
23 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

d_CO[out_ind] = A0;
d_CO[out_ind + n_val_accum] = A1;
d_CO[out_ind + 2 * n_val_accum] = A2;
d_CO[out_ind + 3 * n_val_accum] = A3;
d_CO[out_ind + 4 * n_val_accum] = A4;
d_CO[out_ind + 5 * n_val_accum] = A5;
d_CO[out_ind + 6 * n_val_accum] = A6;
d_CO[out_ind + 7 * n_val_accum] = A7;
d_CO[out_ind + 8 * n_val_accum] = A8;
d_CO[out_ind + 9 * n_val_accum] = A9;
d_CO[out_ind + 10 * n_val_accum] = A10;
d_CO[out_ind + 11 * n_val_accum] = A11;
d_CO[out_ind + 12 * n_val_accum] = A12;
d_CO[out_ind + 13 * n_val_accum] = A13;
d_CO[out_ind + 14 * n_val_accum] = A14;
d_CO[out_ind + 15 * n_val_accum] = A15;
d_CO[out_ind + 16 * n_val_accum] = A16;
d_CO[out_ind + 17 * n_val_accum] = A17;
d_CO[out_ind + 18 * n_val_accum] = A18;
d_CO[out_ind + 19 * n_val_accum] = A19;
d_CO[out_ind + 20 * n_val_accum] = A20;
d_CO[out_ind + 21 * n_val_accum] = A21;
d_CO[out_ind + 22 * n_val_accum] = A22;
}