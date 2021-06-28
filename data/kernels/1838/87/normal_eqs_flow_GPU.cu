#include "includes.h"
__global__ void normal_eqs_flow_GPU(float *d_CO, const float2 *d_flow_compact, const float *d_Zbuffer_flow_compact, const int *d_ind_flow_Zbuffer, float fx, float fy, float ox, float oy, int n_rows, int n_cols, const int *d_n_values_flow, const int *d_start_ind_flow) {

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

/************************/
/* evaluate constraints */
/************************/

// unique values A-matrix
A0 += (disp * disp * fx * fx);
A1 += (-disp * disp * x * fx);
A2 += (-disp * x * y);
A3 += (disp * fx * fx + disp * x * x);
A4 += (-disp * y * fx);
A5 += (-disp * disp * y * fy);
A6 += (-disp * fy * fy - disp * y * y); //!!!!
A7 += (disp * x * fy);
A8 += (disp * disp * x * x + disp * disp * y * y);
A9 += (disp * x * x * y / fx + disp * y * fy + disp * y * y * y / fy);
A10 += (-disp * x * fx - disp * x * x * x / fx - disp * x * y * y / fy);
A11 += (x * x * y * y / (fx * fx) + fy * fy + 2.0f * y * y +
y * y * y * y / (fy * fy));
A12 += (-2.0f * x * y - x * x * x * y / (fx * fx) -
x * y * y * y / (fy * fy));
A13 += (x * y * y / fx - x * fy - x * y * y / fy);
A14 += (fx * fx + 2.0f * x * x + x * x * x * x / (fx * fx) +
x * x * y * y / (fy * fy));
A15 += (-y * fx - x * x * y / fx + x * x * y / fy);
A16 += (x * x + y * y);

// B-vector

A17 += (disp * u.x * fx);
A18 += (disp * u.y * fy);
A19 += (-disp * x * u.x - disp * y * u.y);
A20 += (-x * y * u.x / fx - u.y * fy - u.y * y * y / fy);
A21 += (u.x * fx + x * x * u.x / fx + x * y * u.y / fy);
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