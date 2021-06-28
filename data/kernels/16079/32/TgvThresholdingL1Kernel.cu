#include "includes.h"
__global__ void TgvThresholdingL1Kernel(float2* Tp, float* u_, float* Iu, float* Iz, float lambda, float tau, float* eta_u, float* u, float* us, int width, int height, int stride)
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
us[pos] = 0.0f;// u[pos];
}
else {
//int right = (ix + 1) + iy * stride;
//int down = ix + (iy + 1) * stride;
int left = (ix - 1) + iy * stride;
int up = ix + (iy - 1) * stride;

//div_p = dxm(Tp(:, : , 1)) + dym(Tp(:, : , 2));
float div_p;
float dxmTp, dymTp;
if ((ix - 1) >= 0)
dxmTp = Tp[pos].x - Tp[left].x;
else if (ix == width - 1)
dxmTp = -Tp[left].x;
else
dxmTp = Tp[pos].x;

if ((iy - 1) >= 0)
dymTp = Tp[pos].y - Tp[up].y;
else if (iy == height - 1)
dymTp = -Tp[up].y;
else
dymTp = Tp[pos].y;

div_p = dxmTp + dymTp;

//tau_eta_u = tau. / eta_u;
float tau_eta_u;
if (eta_u[pos] == 0) {
tau_eta_u = tau;
}
else {
tau_eta_u = tau / eta_u[pos];
}

// Thresholding
float uhat = u_[pos] + tau_eta_u * div_p;

float dun = (uhat - u[pos]);

float Ius = Iu[pos];
float rho = Ius * dun + Iz[pos];

float upper = lambda * tau_eta_u*(Ius*Ius);
float lower = -lambda * tau_eta_u*(Ius*Ius);
float du;

if ((rho <= upper) && (rho >= lower)) {
if (Ius == 0) {
du = dun;
}
else {
du = dun - rho / Ius;
}
}
else if (rho < lower) {
du = dun + lambda * tau_eta_u*Ius;
}
else if (rho > upper) {
du = dun - lambda * tau_eta_u*Ius;
}

us[pos] = u[pos] + du;
}
}
}