#include "includes.h"
/*
* DataReader.cpp
*
*  Created on: 06.06.2016
*      Author: Sebastian Reinhart
*/



__global__ void getRelevantMeas(cartesian_segment* carSegs, laserdata_cartesian* d_laser, unsigned long long* dist)
{
int index = blockIdx.x*3;
d_laser[index] = carSegs[blockIdx.x].measures[0];
d_laser[index+2] = carSegs[blockIdx.x].measures[carSegs[blockIdx.x].numberOfMeasures-1];
unsigned long long tmp;
if(threadIdx.x < carSegs[blockIdx.x].numberOfMeasures)
{
float x = carSegs[blockIdx.x].measures[threadIdx.x].x;
float y = carSegs[blockIdx.x].measures[threadIdx.x].y;
tmp = sqrtf(x*x + y*y)*10000;
atomicMin(&(dist[blockIdx.x]), tmp);
__syncthreads();
if(dist[blockIdx.x] == tmp)
{
d_laser[index+1] = carSegs[blockIdx.x].measures[threadIdx.x];
}
}
}