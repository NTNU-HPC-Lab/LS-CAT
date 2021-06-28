#include "includes.h"
__global__ void devicetodevicecopy(double *dphi, double *dpsix, double *dpsiy, double *mphi, double *mpsix, double *mpsiy, unsigned int nx, unsigned int TileSize)
{
unsigned int bx = blockIdx.x;
unsigned int by = blockIdx.y;

unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;

unsigned int index_x = bx * TileSize + tx;
unsigned int index_y = by * TileSize + ty;

unsigned int indexToWrite = index_y * nx + index_x;

mphi[indexToWrite] = dphi[indexToWrite];
mpsix[indexToWrite] = dpsix[indexToWrite];
mpsiy[indexToWrite] = dpsiy[indexToWrite];
}