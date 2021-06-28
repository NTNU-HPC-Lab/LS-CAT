#include "includes.h"
/**
*
* Copyright (C) Tyler Hackett 2016
*
* CUDA Triangle Counter
*
* A quickly-written program to determine all possible combinations of
* valid triangles from a grid, allowing for certain coordinates of the
* grid to be marked as unusable.
*
* main.cu
*
* */



__global__ void countTriangles(uint2 *validPoints, int *count)
{
/* Only allow operations on blocks where x < y < z, to prevent repeat triangles*/
if (blockIdx.x > blockIdx.y || blockIdx.y > blockIdx.z || blockIdx.x > blockIdx.z)
return;

uint2 x, y, z;
x = validPoints[blockIdx.x];
y = validPoints[blockIdx.y];
z = validPoints[blockIdx.z];

/*Check if the points are coplanar.*/
if ((x.x == y.x || x.y == y.y) && (y.x == z.x || y.y == z.y) && (x.x == z.x || x.y == z.y))
return;
/*Check for any coincident points.*/
if ((x.x == y.x && x.y == y.y) || (y.x == z.x && y.y == z.y) || (x.x == z.x && x.y == z.y))
return;

/*If the thread makes it this far, then we have a triangle that obeys the laws of geometry!*/
atomicAdd(count, 1);
}