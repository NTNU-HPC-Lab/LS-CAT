#include "includes.h"
__global__ void normal_eqs_flow_multicam_GPU( float *d_CO, float2 *d_flow_compact, float *d_Zbuffer_flow_compact, int *d_ind_flow_Zbuffer, const float *d_focal_length, const float *d_nodal_point_x, const float *d_nodal_point_y, const int *d_n_rows, const int *d_n_cols, const int *d_n_values_flow, const int *d_start_ind_flow, const int *d_pixel_ind_offset) {
int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
// multiple of blocksize

int n_flow = d_n_values_flow[blockIdx.y];
int n_accum = (int)ceilf((float)n_flow / (float)n_val_accum);
int start_ind = d_start_ind_flow[blockIdx.y];

float f = d_focal_length[blockIdx.y];
float ox = d_nodal_point_x[blockIdx.y];
float oy = d_nodal_point_y[blockIdx.y];
int n_rows = d_n_rows[blockIdx.y];
int n_cols = d_n_cols[blockIdx.y];
int pixel_ind_offset = d_pixel_ind_offset[blockIdx.y];

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
int pixel_ind = d_ind_flow_Zbuffer[in_ind + start_ind] - pixel_ind_offset;
bool is_ar_flow = (pixel_ind >= (n_rows * n_cols));
pixel_ind -= (int)is_ar_flow * n_rows * n_cols;

float y = floorf(__fdividef((float)pixel_ind, n_cols));
float x = (float)pixel_ind - y * n_cols;

x = x - ox;
y = y - oy;

// flip y axis
y = -y;
u.y = -u.y;

/************************/
/* evaluate constraints */
/************************/

// unique values A-matrix
A0 += (disp * disp * f * f);
A1 += (-disp * disp * x * f);
A2 += (-disp * x * y);
A3 += (disp * f * f + disp * x * x);
A4 += (-disp * y * f);
A5 += (-disp * disp * y * f);
A6 += (-disp * f * f - disp * y * y);
A7 += (disp * x * f);
A8 += (disp * disp * x * x + disp * disp * y * y);
A9 += (disp * x * x * y / f + disp * y * f + disp * y * y * y / f);
A10 += (-disp * x * f - disp * x * x * x / f - disp * x * y * y / f);
A11 += (x * x * y * y / (f * f) + f * f + 2.0f * y * y +
y * y * y * y / (f * f));
A12 +=
(-2.0f * x * y - x * x * x * y / (f * f) - x * y * y * y / (f * f));
A13 += (-x * f);
A14 += (f * f + 2.0f * x * x + x * x * x * x / (f * f) +
x * x * y * y / (f * f));
A15 += (-y * f);
A16 += (x * x + y * y);

// B-vector

A17 += (disp * u.x * f);
A18 += (disp * u.y * f);
A19 += (-disp * x * u.x - disp * y * u.y);
A20 += (-x * y * u.x / f - u.y * f - u.y * y * y / f);
A21 += (u.x * f + x * x * u.x / f + x * y * u.y / f);
A22 += (-y * u.x + x * u.y);
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