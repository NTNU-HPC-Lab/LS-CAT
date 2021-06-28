#include "includes.h"
__device__ static float disp_absolute_residual(float Xd, float Yd, float Zd, float Xm, float Ym, float Zm, float nx, float ny, float nz, float T0, float T1, float T2, float R0, float R1, float R2, float fx, float b) {
float r = -Xd * nx + Xm * nx - Yd * ny + Ym * ny - Zd * nz + Zm * nz +
nx * T0 + ny * T1 + nz * T2 + Xm * ny * R2 - Xm * nz * R1 -
Ym * nx * R2 + Ym * nz * R0 + Zm * nx * R1 - Zm * ny * R0;

// weight to convert distance units to pixels
r *= fx * b / (Zm * Zm);

return fabsf(r);
}
__global__ void disp_absolute_residual_scalable_GPU( float *d_abs_res, const float *d_disparity_compact, const float4 *d_Zbuffer_normals_compact, const int *d_ind_disparity_Zbuffer, const unsigned int *d_valid_disparity_Zbuffer, float fx, float fy, float ox, float oy, float b, int n_cols, int n_valid_disparity_Zbuffer, const int *d_offset_ind, const int *d_segment_translation_table, float w_disp, const float *d_dTR) {

int ind = blockDim.x * blockIdx.x + threadIdx.x;

if (ind < n_valid_disparity_Zbuffer) {

// determine current segment
int segment = d_segment_translation_table[d_valid_disparity_Zbuffer[ind]];

// fetch disparity, Zbuffer and normal from global memory
float disp = d_disparity_compact[ind];
float4 tmp = d_Zbuffer_normals_compact[ind];
float Zbuffer = tmp.x;
float nx = tmp.y;
float ny = tmp.z;
float nz = tmp.w;

// compute coordinates
int pixel_ind = d_ind_disparity_Zbuffer[ind];

float y = floorf(__fdividef((float)pixel_ind, n_cols));
float x = (float)pixel_ind - y * n_cols;

x = __fdividef((x - ox), fx);
y = __fdividef((y - oy), fy);

// reconstruct 3D point from disparity
float Zd = -(fx * b) / disp; // arbitrary use of fx for now
float Xd = x * Zd;
float Yd = y * Zd;

// reconstruct 3D point from model
float Zm = Zbuffer;
float Xm = x * Zm;
float Ym = y * Zm;

// compute absolute residual (weighted by disparity vs flow importance)
int ind_out = ind + d_offset_ind[segment];
int s6 = segment * 6;
d_abs_res[ind_out] =
w_disp * disp_absolute_residual(Xd, Yd, Zd, Xm, Ym, Zm, nx, ny, nz,
d_dTR[s6], d_dTR[s6 + 1], d_dTR[s6 + 2],
d_dTR[s6 + 3], d_dTR[s6 + 4],
d_dTR[s6 + 5], fx, b);
}
}