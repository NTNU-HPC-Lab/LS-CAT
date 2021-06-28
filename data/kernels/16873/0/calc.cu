#include "includes.h"

#define GravConst 6.674e-11
#define EPS 1e-6
#define POINTS_SIZE 10

//struct Point {
//    float x, y, z;
//    float vx = 0, vy = 0, vz = 0;
//    float ax = 0, ay = 0, az = 0;
//    float m;
//};


__device__ inline float sqr(float x) { return x * x; }
__global__ void calc(float* points, float* resPoints, int dt, int size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;

int tix = threadIdx.x;
int steps = gridDim.x;

float fx,fy,fz;

float x, y, z, m, vx, vy, vz, ax, ay, az;

if (idx < size) {
x = points[idx * POINTS_SIZE + 0];
y = points[idx * POINTS_SIZE + 1];
z = points[idx * POINTS_SIZE + 2];
m = points[idx * POINTS_SIZE + 3];
vx = points[idx * POINTS_SIZE + 4];
vy = points[idx * POINTS_SIZE + 5];
vz = points[idx * POINTS_SIZE + 6];
ax = points[idx * POINTS_SIZE + 7];
ay = points[idx * POINTS_SIZE + 8];
az = points[idx * POINTS_SIZE + 9];

for (int iteration = 0; iteration < steps; iteration++) {

__shared__ float cached_points[128 * POINTS_SIZE];

if (iteration * 128 + tix < size)
for (int i = 0; i < POINTS_SIZE; i++)
cached_points[tix * POINTS_SIZE] = points[(iteration * 128 + tix) * POINTS_SIZE + i];
__syncthreads();

if (idx < size) {
fx = fy = fz = 0;
for (int i = 0; i < 128; i++) {
if (iteration * 128 + i < size && iteration * 128 + i != idx) {
float dx = cached_points[i * POINTS_SIZE] - x;
float dy = cached_points[i * POINTS_SIZE + 1] - y;
float dz = cached_points[i * POINTS_SIZE + 2] - z;
float dist = sqrt(dx * dx + dy * dy + dz * dz);
float F = (GravConst * m * cached_points[i * POINTS_SIZE + 3]) / (dist * dist + 0.001f * 0.001f);
fx += F * dx / dist;
fy += F * dy / dist;
fz += F * dz / dist;
}
__syncthreads();
}
resPoints[idx * POINTS_SIZE] = x + vx * dt + (ax * sqr(dt)) / 2;
resPoints[idx * POINTS_SIZE + 1] = y + vy * dt + (ay * sqr(dt)) / 2;
resPoints[idx * POINTS_SIZE + 2] = z + vz * dt + (az * sqr(dt)) / 2;

resPoints[idx * POINTS_SIZE + 3] = m;

resPoints[idx * POINTS_SIZE + 4] = vx + ax * dt;
resPoints[idx * POINTS_SIZE + 5] = vy + ay * dt;
resPoints[idx * POINTS_SIZE + 6] = vz + az * dt;

resPoints[idx * POINTS_SIZE + 7] = fx / m;
resPoints[idx * POINTS_SIZE + 8] = fy / m;
resPoints[idx * POINTS_SIZE + 9] = fz / m;
}
}
}







/*if (idx < size) {
float* forcesArr = forcesCalc(points, idx, size);
resPoints[idx] = points[idx] + points[idx + 4] * dt + (points[idx + 7] * sqr(dt)) / 2;
resPoints[idx + 1] = points[idx + 1] + points[idx + 5] * dt + (points[idx + 8] * sqr(dt)) / 2;
resPoints[idx + 2] = points[idx + 2] + points[idx + 6] * dt + (points[idx + 9] * sqr(dt)) / 2;

resPoints[idx + 3] = points[idx + 3];

resPoints[idx + 4] = points[idx + 4] + points[idx + 7] * dt;
resPoints[idx + 5] = points[idx + 5] + points[idx + 8] * dt;
resPoints[idx + 6] = points[idx + 6] + points[idx + 9] * dt;

resPoints[idx + 7] = forcesArr[0] / points[idx + 3];
resPoints[idx + 8] = forcesArr[1] / points[idx + 3];
resPoints[idx + 9] = forcesArr[2] / points[idx + 3];
}*/
}