#include "includes.h"
__global__ void getRelevantMeas(cartesian_segment* carSegs, laserdata_cartesian* d_laser, unsigned long long* dist)
{
int index = blockIdx.x*3;
//first and last entry is trivial
d_laser[index] = carSegs[blockIdx.x].measures[0];
d_laser[index+2] = carSegs[blockIdx.x].measures[carSegs[blockIdx.x].numberOfMeasures-1];
unsigned long long tmp;
//check whether thread is out of bounds
if(threadIdx.x < carSegs[blockIdx.x].numberOfMeasures)
{
//compute distance for current position
float x = carSegs[blockIdx.x].measures[threadIdx.x].x;
float y = carSegs[blockIdx.x].measures[threadIdx.x].y;
tmp = sqrtf(x*x + y*y)*10000;
//write to shared memory
atomicMin(&(dist[blockIdx.x]), tmp);
__syncthreads();
if(dist[blockIdx.x] == tmp)
{
//own position is neareast -> write to out array
d_laser[index+1] = carSegs[blockIdx.x].measures[threadIdx.x];
}
}
}