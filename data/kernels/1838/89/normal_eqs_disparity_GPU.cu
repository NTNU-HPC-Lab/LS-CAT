#include "includes.h"
__global__ void normal_eqs_disparity_GPU(float *d_CD, const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact, const int *d_ind_disparity_Zbuffer, float fx, float fy, float ox, float oy, float b, int n_cols, const int *d_n_values_disparity, const int *d_start_ind_disparity, float w_disp) {

int n_val_accum = gridDim.x * blockDim.x; // _MAX_N_VAL_ACCUM may not be
// multiple of blocksize

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

float Zd = -(fx * b) / disp; // arbitrary conversion for now using fx
float Xd = x * Zd;
float Yd = y * Zd;

// reconstruct 3D point from model

float Zm = Zbuffer;
float Xm = x * Zm;
float Ym = y * Zm;

// weight the constraint according to (fx*b)/(Zm*Zm) to convert
// from distance- (mm) to image-units (pixel)
float w2 = fx * b / (Zm * Zm);
w2 *= w2;

/************************/
/* evaluate constraints */
/************************/

// unique values A-matrix

A0 += w2 * (nx * nx);
A1 += w2 * (nx * ny);
A2 += w2 * (nx * nz);
A3 += w2 * (Ym * nx * nz - Zm * nx * ny);
A4 += w2 * (Zm * (nx * nx) - Xm * nx * nz);
A5 += w2 * (-Ym * (nx * nx) + Xm * nx * ny);

A6 += w2 * (ny * ny);
A7 += w2 * (ny * nz);
A8 += w2 * (-Zm * (ny * ny) + Ym * ny * nz);
A9 += w2 * (-Xm * ny * nz + Zm * nx * ny);
A10 += w2 * (Xm * (ny * ny) - Ym * nx * ny);

A11 += w2 * (nz * nz);
A12 += w2 * (Ym * (nz * nz) - Zm * ny * nz);
A13 += w2 * (-Xm * (nz * nz) + Zm * nx * nz);
A14 += w2 * (Xm * ny * nz - Ym * nx * nz);

A15 += w2 * ((Ym * Ym) * (nz * nz) + (Zm * Zm) * (ny * ny) -
Ym * Zm * ny * nz * 2.0f);
A16 += w2 * (-Xm * Ym * (nz * nz) - (Zm * Zm) * nx * ny +
Xm * Zm * ny * nz + Ym * Zm * nx * nz);
A17 += w2 * (-Xm * Zm * (ny * ny) - (Ym * Ym) * nx * nz +
Xm * Ym * ny * nz + Ym * Zm * nx * ny);

A18 += w2 * ((Xm * Xm) * (nz * nz) + (Zm * Zm) * (nx * nx) -
Xm * Zm * nx * nz * 2.0f);
A19 += w2 * (-Ym * Zm * (nx * nx) - (Xm * Xm) * ny * nz +
Xm * Ym * nx * nz + Xm * Zm * nx * ny);

A20 += w2 * ((Xm * Xm) * (ny * ny) + (Ym * Ym) * (nx * nx) -
Xm * Ym * nx * ny * 2.0f);

// B-vector

A21 += w2 * (Xd * (nx * nx) - Xm * (nx * nx) + Yd * nx * ny -
Ym * nx * ny + Zd * nx * nz - Zm * nx * nz);
A22 += w2 * (Yd * (ny * ny) - Ym * (ny * ny) + Xd * nx * ny -
Xm * nx * ny + Zd * ny * nz - Zm * ny * nz);
A23 += w2 * (Zd * (nz * nz) - Zm * (nz * nz) + Xd * nx * nz -
Xm * nx * nz + Yd * ny * nz - Ym * ny * nz);
A24 += w2 *
(-Yd * Zm * (ny * ny) + Ym * Zd * (nz * nz) + Ym * Zm * (ny * ny) -
Ym * Zm * (nz * nz) - (Ym * Ym) * ny * nz + (Zm * Zm) * ny * nz +
Xd * Ym * nx * nz - Xm * Ym * nx * nz - Xd * Zm * nx * ny +
Yd * Ym * ny * nz + Xm * Zm * nx * ny - Zd * Zm * ny * nz);
A25 += w2 *
(Xd * Zm * (nx * nx) - Xm * Zd * (nz * nz) - Xm * Zm * (nx * nx) +
Xm * Zm * (nz * nz) + (Xm * Xm) * nx * nz - (Zm * Zm) * nx * nz -
Xd * Xm * nx * nz - Xm * Yd * ny * nz + Xm * Ym * ny * nz +
Yd * Zm * nx * ny - Ym * Zm * nx * ny + Zd * Zm * nx * nz);
A26 += w2 *
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