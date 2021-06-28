#include "includes.h"
__device__ static float disp_absolute_residual(float Xd, float Yd, float Zd, float Xm, float Ym, float Zm, float nx, float ny, float nz, float T0, float T1, float T2, float R0, float R1, float R2, float fx, float b) {
float r = -Xd * nx + Xm * nx - Yd * ny + Ym * ny - Zd * nz + Zm * nz +
nx * T0 + ny * T1 + nz * T2 + Xm * ny * R2 - Xm * nz * R1 -
Ym * nx * R2 + Ym * nz * R0 + Zm * nx * R1 - Zm * ny * R0;

// weight to convert distance units to pixels
r *= fx * b / (Zm * Zm);

return fabsf(r);
}
__global__ void normal_eqs_disparity_weighted_GPU( float *d_CD, const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact, const int *d_ind_disparity_Zbuffer, float fx, float fy, float ox, float oy, float b, int n_cols, const int *d_n_values_disparity, const int *d_start_ind_disparity, const float *d_abs_res_scales, float w_disp, const float *d_dTR) {

int n_val_accum =
gridDim.x * blockDim.x; // n_val_accum may not be multiple of blocksize

int n_disparity = d_n_values_disparity[blockIdx.y];
int n_accum = (int)ceilf((float)n_disparity / (float)n_val_accum);
int start_ind = d_start_ind_disparity[blockIdx.y];

// initialize accumulators

float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f, A4 = 0.0f, A5 = 0.0f,
A6 = 0.0f, A7 = 0.0f, A8 = 0.0f, A9 = 0.0f, A10 = 0.0f, A11 = 0.0f,
A12 = 0.0f, A13 = 0.0f, A14 = 0.0f, A15 = 0.0f, A16 = 0.0f, A17 = 0.0f,
A18 = 0.0f, A19 = 0.0f, A20 = 0.0f, A21 = 0.0f, A22 = 0.0f, A23 = 0.0f,
A24 = 0.0f, A25 = 0.0f, A26 = 0.0f;

for (int in_ind = blockDim.x * blockIdx.x * n_accum + threadIdx.x;
in_ind < blockDim.x * (blockIdx.x + 1) * n_accum; in_ind += blockDim.x) {

if (in_ind < n_disparity) { // is this a valid sample?

// fetch disparity, Zbuffer and normal from global memory
float disp = d_disparity_compact[in_ind + start_ind];
float4 tmp = d_Zbuffer_normals_compact[in_ind + start_ind];
float Zbuffer = tmp.x;
float nx = tmp.y;
float ny = tmp.z;
float nz = tmp.w;

// compute coordinates
int pixel_ind = d_ind_disparity_Zbuffer[in_ind + start_ind];

float y = floorf(__fdividef((float)pixel_ind, n_cols));
float x = (float)pixel_ind - y * n_cols;

x = __fdividef((x - ox), fx);
y = __fdividef((y - oy), fy);

// reconstruct 3D point from disparity

float Zd = -(fx * b) / disp; // arbitrary use of fx
float Xd = x * Zd;
float Yd = y * Zd;

// reconstruct 3D point from model

float Zm = Zbuffer;
float Xm = x * Zm;
float Ym = y * Zm;

// determine M-estimation weight
// disparity residual weighed by rel. importance disp vs flow
int s6 = blockIdx.y * 6;
float w = w_disp * disp_absolute_residual(
Xd, Yd, Zd, Xm, Ym, Zm, nx, ny, nz, d_dTR[s6],
d_dTR[s6 + 1], d_dTR[s6 + 2], d_dTR[s6 + 3],
d_dTR[s6 + 4], d_dTR[s6 + 5], fx, b);
w /= d_abs_res_scales[blockIdx.y];
w = (w > 1) ? 0 : (1.0f - 2.0f * w * w + w * w * w * w);

// multiply m estimation weight with distance->pixel conversion weight
// (squared)
w *= (fx * fx * b * b) / (Zm * Zm * Zm * Zm);

/************************/
/* evaluate constraints */
/************************/

// unique values A-matrix

A0 += w * (nx * nx);
A1 += w * (nx * ny);
A2 += w * (nx * nz);
A3 += w * (Ym * nx * nz - Zm * nx * ny);
A4 += w * (Zm * (nx * nx) - Xm * nx * nz);
A5 += w * (-Ym * (nx * nx) + Xm * nx * ny);

A6 += w * (ny * ny);
A7 += w * (ny * nz);
A8 += w * (-Zm * (ny * ny) + Ym * ny * nz);
A9 += w * (-Xm * ny * nz + Zm * nx * ny);
A10 += w * (Xm * (ny * ny) - Ym * nx * ny);

A11 += w * (nz * nz);
A12 += w * (Ym * (nz * nz) - Zm * ny * nz);
A13 += w * (-Xm * (nz * nz) + Zm * nx * nz);
A14 += w * (Xm * ny * nz - Ym * nx * nz);

A15 += w * ((Ym * Ym) * (nz * nz) + (Zm * Zm) * (ny * ny) -
Ym * Zm * ny * nz * 2.0f);
A16 += w * (-Xm * Ym * (nz * nz) - (Zm * Zm) * nx * ny +
Xm * Zm * ny * nz + Ym * Zm * nx * nz);
A17 += w * (-Xm * Zm * (ny * ny) - (Ym * Ym) * nx * nz +
Xm * Ym * ny * nz + Ym * Zm * nx * ny);

A18 += w * ((Xm * Xm) * (nz * nz) + (Zm * Zm) * (nx * nx) -
Xm * Zm * nx * nz * 2.0f);
A19 += w * (-Ym * Zm * (nx * nx) - (Xm * Xm) * ny * nz +
Xm * Ym * nx * nz + Xm * Zm * nx * ny);

A20 += w * ((Xm * Xm) * (ny * ny) + (Ym * Ym) * (nx * nx) -
Xm * Ym * nx * ny * 2.0f);

// B-vector

A21 += w * (Xd * (nx * nx) - Xm * (nx * nx) + Yd * nx * ny -
Ym * nx * ny + Zd * nx * nz - Zm * nx * nz);
A22 += w * (Yd * (ny * ny) - Ym * (ny * ny) + Xd * nx * ny -
Xm * nx * ny + Zd * ny * nz - Zm * ny * nz);
A23 += w * (Zd * (nz * nz) - Zm * (nz * nz) + Xd * nx * nz -
Xm * nx * nz + Yd * ny * nz - Ym * ny * nz);
A24 += w *
(-Yd * Zm * (ny * ny) + Ym * Zd * (nz * nz) + Ym * Zm * (ny * ny) -
Ym * Zm * (nz * nz) - (Ym * Ym) * ny * nz + (Zm * Zm) * ny * nz +
Xd * Ym * nx * nz - Xm * Ym * nx * nz - Xd * Zm * nx * ny +
Yd * Ym * ny * nz + Xm * Zm * nx * ny - Zd * Zm * ny * nz);
A25 +=
w * (Xd * Zm * (nx * nx) - Xm * Zd * (nz * nz) - Xm * Zm * (nx * nx) +
Xm * Zm * (nz * nz) + (Xm * Xm) * nx * nz - (Zm * Zm) * nx * nz -
Xd * Xm * nx * nz - Xm * Yd * ny * nz + Xm * Ym * ny * nz +
Yd * Zm * nx * ny - Ym * Zm * nx * ny + Zd * Zm * nx * nz);
A26 += w *
(-Xd * Ym * (nx * nx) + Xm * Yd * (ny * ny) + Xm * Ym * (nx * nx) -
Xm * Ym * (ny * ny) - (Xm * Xm) * nx * ny + (Ym * Ym) * nx * ny +
Xd * Xm * nx * ny - Yd * Ym * nx * ny + Xm * Zd * ny * nz -
Xm * Zm * ny * nz - Ym * Zd * nx * nz + Ym * Zm * nx * nz);
}
}

/**************************/
/* write out accumulators */
/**************************/

int out_ind =
27 * n_val_accum * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

w_disp *= w_disp; // weight relative to flow

d_CD[out_ind] = w_disp * A0;
d_CD[out_ind + n_val_accum] = w_disp * A1;
d_CD[out_ind + 2 * n_val_accum] = w_disp * A2;
d_CD[out_ind + 3 * n_val_accum] = w_disp * A3;
d_CD[out_ind + 4 * n_val_accum] = w_disp * A4;
d_CD[out_ind + 5 * n_val_accum] = w_disp * A5;
d_CD[out_ind + 6 * n_val_accum] = w_disp * A6;
d_CD[out_ind + 7 * n_val_accum] = w_disp * A7;
d_CD[out_ind + 8 * n_val_accum] = w_disp * A8;
d_CD[out_ind + 9 * n_val_accum] = w_disp * A9;
d_CD[out_ind + 10 * n_val_accum] = w_disp * A10;
d_CD[out_ind + 11 * n_val_accum] = w_disp * A11;
d_CD[out_ind + 12 * n_val_accum] = w_disp * A12;
d_CD[out_ind + 13 * n_val_accum] = w_disp * A13;
d_CD[out_ind + 14 * n_val_accum] = w_disp * A14;
d_CD[out_ind + 15 * n_val_accum] = w_disp * A15;
d_CD[out_ind + 16 * n_val_accum] = w_disp * A16;
d_CD[out_ind + 17 * n_val_accum] = w_disp * A17;
d_CD[out_ind + 18 * n_val_accum] = w_disp * A18;
d_CD[out_ind + 19 * n_val_accum] = w_disp * A19;
d_CD[out_ind + 20 * n_val_accum] = w_disp * A20;
d_CD[out_ind + 21 * n_val_accum] = w_disp * A21;
d_CD[out_ind + 22 * n_val_accum] = w_disp * A22;
d_CD[out_ind + 23 * n_val_accum] = w_disp * A23;
d_CD[out_ind + 24 * n_val_accum] = w_disp * A24;
d_CD[out_ind + 25 * n_val_accum] = w_disp * A25;
d_CD[out_ind + 26 * n_val_accum] = w_disp * A26;
}