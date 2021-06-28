#include "includes.h"
__global__ void TgvUpdateDualVariablesTGVKernel(float* u_, float2 *v_, float alpha0, float alpha1, float sigma, float eta_p, float eta_q, float* a, float* b, float*c, float4* grad_v, float2* p, float4* q, int width, int height, int stride)
{
int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row
int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column

float desiredRadius = (float)width / 2.20f;
float halfWidth = (float)width / 2.0f;
float halfHeight = (float)height / 2.0f;
float radius = sqrtf((iy - halfHeight) * (iy - halfHeight) + (ix - halfWidth) * (ix - halfWidth));

if ((iy < height) && (ix < width))
{
int pos = ix + iy * stride;

if (radius >= desiredRadius)
{
p[pos] = make_float2(0.0f, 0.0f);
q[pos] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}
else {
int right = (ix + 1) + iy * stride;
int down = ix + (iy + 1) * stride;
int left = (ix - 1) + iy * stride;
int up = ix + (iy - 1) * stride;

//u_x = dxp(u_) - v_(:, : , 1);
float u_x, u_y;
if ((ix + 1) < width) u_x = u_[right] - u_[pos] - v_[pos].x;
else u_x = u_[pos] - u_[left] - v_[pos].x;
//u_y = dyp(u_) - v_(:, : , 2);
if ((iy + 1) < height) u_y = u_[down] - u_[pos] - v_[pos].y;
else u_y = u_[pos] - u_[up] - v_[pos].y;

//du_tensor_x = a.*u_x + c.*u_y;
float du_tensor_x = a[pos] * u_x + c[pos] * u_y;
//du_tensor_y = c.*u_x + b.*u_y;
float du_tensor_y = c[pos] * u_x + b[pos] * u_y;

//p(:, : , 1) = p(:, : , 1) + alpha1*sigma / eta_p.*du_tensor_x;
p[pos].x = p[pos].x + (alpha1*sigma / eta_p) * du_tensor_x;
//p(:, : , 2) = p(:, : , 2) + alpha1*sigma / eta_p.*du_tensor_y;
p[pos].y = p[pos].y + (alpha1*sigma / eta_p) * du_tensor_y;

//projection
//reprojection = max(1.0, sqrt(p(:, : , 1). ^ 2 + p(:, : , 2). ^ 2));
float reprojection = sqrtf(p[pos].x * p[pos].x + p[pos].y * p[pos].y);
if (reprojection < 1.0f) {
reprojection = 1.0f;
}
//p(:, : , 1) = p(:, : , 1). / reprojection;
p[pos].x = p[pos].x / reprojection;
//p(:, : , 2) = p(:, : , 2). / reprojection;
p[pos].y = p[pos].y / reprojection;

//grad_v(:, : , 1) = dxp(v_(:, : , 1));
if ((ix + 1) < width) grad_v[pos].x = v_[right].x - v_[pos].x;
else grad_v[pos].x = v_[pos].x - v_[left].x;

//grad_v(:, : , 2) = dyp(v_(:, : , 2));
if ((iy + 1) < height) grad_v[pos].y = v_[down].y - v_[pos].y;
else grad_v[pos].y = v_[pos].y - v_[up].y;

//grad_v(:, : , 3) = dyp(v_(:, : , 1));
if ((iy + 1) < height) grad_v[pos].z = v_[down].x - v_[pos].x;
else grad_v[pos].z = v_[pos].x - v_[up].x;

//grad_v(:, : , 4) = dxp(v_(:, : , 2));
if ((ix + 1) < width) grad_v[pos].w = v_[right].y - v_[pos].y;
else grad_v[pos].w = v_[pos].y - v_[left].y;

//q = q + alpha0*sigma / eta_q.*grad_v;
float ase = alpha0 * sigma / eta_q;
float4 qpos;
qpos.x = q[pos].x + ase * grad_v[pos].x;
qpos.y = q[pos].y + ase * grad_v[pos].y;
qpos.z = q[pos].z + ase * grad_v[pos].z;
qpos.w = q[pos].w + ase * grad_v[pos].w;

//reproject = max(1.0, sqrt(q(:, : , 1). ^ 2 + q(:, : , 2). ^ 2 + q(:, : , 3). ^ 2 + q(:, : , 4). ^ 2));
float reproject = sqrtf(qpos.x * qpos.x + qpos.y * qpos.y + qpos.z * qpos.z + qpos.w * qpos.w);
if (reproject < 1.0f) {
reproject = 1.0f;
}
//q(:, : , 1) = q(:, : , 1). / reproject;
q[pos].x = qpos.x / reproject;
//q(:, : , 2) = q(:, : , 2). / reproject;
q[pos].y = qpos.y / reproject;
//q(:, : , 3) = q(:, : , 3). / reproject;
q[pos].z = qpos.z / reproject;
//q(:, : , 4) = q(:, : , 4). / reproject;
q[pos].w = qpos.w / reproject;
}
}
}