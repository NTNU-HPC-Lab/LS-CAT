#include "includes.h"
__device__ unsigned int locationAlgo(double *x, double xadv, unsigned int nx)
{
unsigned int location = 0;
while (x[location] < xadv && location < nx)
location++;
if(location == 0)
return location;
else
return location-1;
}
__global__ void find_advection_point_location_cuda(double *x, double *y, double *xadv, double *yadv, unsigned int nx, unsigned int ny, unsigned int *cellx, unsigned int *celly, unsigned int *tracker, double xlim1, double xlim2, double ylim1, double ylim2, unsigned int TileSize)
{
unsigned int bx = blockIdx.x;
unsigned int by = blockIdx.y;

unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;

unsigned int index_x = bx * TileSize + tx;
unsigned int index_y = by * TileSize + ty;

unsigned int indexToWrite = index_y * nx + index_x;

bool xoutofbounds = false;
bool youtofbounds = false;

if(!((xadv[indexToWrite] > xlim1) && (xadv[indexToWrite] < xlim2)))
xoutofbounds = true;
if(!((yadv[indexToWrite] > ylim1) && (yadv[indexToWrite] < ylim2)))
youtofbounds = true;

if(!xoutofbounds && !youtofbounds)
{
tracker[indexToWrite] = 1;
cellx[indexToWrite] = locationAlgo(x,xadv[indexToWrite],nx);
celly[indexToWrite] = locationAlgo(y,yadv[indexToWrite],ny);
}
else
if(!xoutofbounds && youtofbounds)
{
tracker[indexToWrite] = 2;
cellx[indexToWrite] = locationAlgo(x,xadv[indexToWrite],nx);
if(yadv[indexToWrite] <= ylim1)
celly[indexToWrite] = 0;
else
if(yadv[indexToWrite] >= ylim2)
celly[indexToWrite] = ny-2;
}
else
if(xoutofbounds && !youtofbounds)
{
tracker[indexToWrite] = 3;
celly[indexToWrite] = locationAlgo(y,yadv[indexToWrite],ny);
if(xadv[indexToWrite] <= xlim1)
cellx[indexToWrite] = 0;
else
if(xadv[indexToWrite] >= xlim2)
cellx[indexToWrite] = nx-2;
}
else
if(xoutofbounds && youtofbounds)
tracker[indexToWrite] = 4;
}