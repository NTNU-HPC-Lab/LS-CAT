#include "includes.h"
__global__ void kernelGenerateTriangles(int *voronoiPtr, short2 *patternPtr, int3 *ctriangles, int *offset, int width, int min, int max) {
int x = blockIdx.x * blockDim.x + threadIdx.x;

if (x < min || x >= max)
return ;

int xwidth = x * width;
short2 pattern = patternPtr[xwidth + min];
int i0, i1, i2, i3;
int3 *pT = &ctriangles[offset[x-1]];

// Jump through all voronoi vertices in a texture row
while (pattern.y > 0 && pattern.y < max) {
i0 = voronoiPtr[xwidth + pattern.y];
i1 = voronoiPtr[xwidth + pattern.y + 1];
i2 = voronoiPtr[xwidth + width + pattern.y + 1];
i3 = voronoiPtr[xwidth + width + pattern.y];

if (pattern.x == 0) *pT = make_int3(i3, i1, i2);
if (pattern.x == 1) *pT = make_int3(i0, i2, i3);
if (pattern.x == 2) *pT = make_int3(i1, i3, i0);
if (pattern.x == 3) *pT = make_int3(i2, i0, i1);
if (pattern.x == 4) {
// Generate 2 triangles.
// Since the hole is convex, no need to do CCW test
*pT = make_int3(i2, i0, i1); pT++;
*pT = make_int3(i3, i0, i2);
}

pattern = patternPtr[xwidth + pattern.y + 1];
pT++;
}
}