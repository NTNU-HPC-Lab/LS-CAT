#include "includes.h"
__global__ void get_edgemean_kernal(const float* data, float* edgemean, const int nx, const int ny, const int nz)
{
int di = 0;
float edge_sum = 0;
float edge_mean = 0;
size_t nxy = nx * ny;
if (nz == 1) {
for (int i = 0, j = (ny - 1) * nx; i < nx; ++i, ++j) {
edge_sum += data[i] + data[j];
}
for (size_t i = 0, j = nx - 1; i < nxy; i += nx, j += nx) {
edge_sum += data[i] + data[j];
}
edge_mean = (float)edge_sum / (nx * 2 + ny * 2);
}
else {
if (nx == ny && nx == nz * 2 - 1) {
for (size_t j = (nxy * (nz - 1)); j < nxy * nz; ++j, ++di) {
edge_sum += data[j];
}
}
else {
for (size_t i = 0, j = (nxy * (nz - 1)); i < nxy; ++i, ++j, ++di) {
edge_sum += data[i] + data[j];
}
}

int nxy2 = nx * (ny - 1);
for (int k = 1; k < nz - 1; ++k) {
size_t k2 = k * nxy;
size_t k3 = k2 + nxy2;
for (int i = 0; i < nx; ++i, ++di) {
edge_sum += data[i + k2] + data[i + k3];
}
}
for (int k = 1; k < nz - 1; ++k) {
size_t k2 = k * nxy;
size_t k3 = nx - 1 + k2;
for (int i = 1; i < ny - 1; ++i, ++di) {
edge_sum += data[i * nx + k2] + data[i * nx + k3];
}
}

edge_mean = (float)edge_sum / (di * 2);
}
*edgemean = edge_mean;
}