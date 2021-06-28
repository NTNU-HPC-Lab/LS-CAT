#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void pathAdjacencyKernel(int noTransitions, int noSegments, float* XY1, float* XY2, float* X4_X3, float* Y4_Y3, float* X2_X1, float* Y2_Y1, int* adjacency) {

int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int idx = blockId * blockDim.x + threadIdx.x;

if (idx < noTransitions*noSegments) {
int seg1 = idx/noSegments;
int seg2 = idx - seg1*noSegments;

float Y1_Y3 = XY1[seg1 + noTransitions] - XY2[seg2 + noSegments];
float X1_X3 = XY1[seg1] - XY2[seg2];

float numa = X4_X3[seg2]*Y1_Y3 - Y4_Y3[seg2]*X1_X3;
float numb = X2_X1[seg1]*Y1_Y3 - Y2_Y1[seg1]*X1_X3;
float deno = Y4_Y3[seg2]*X2_X1[seg1] - X4_X3[seg2]*Y2_Y1[seg1];

float u_a = numa/deno;
float u_b = numb/deno;

adjacency[idx] = (int)((u_a >= 0.0) && (u_a <= 1.0) && (u_b >= 0.0)
&& (u_b <= 1.0));
}
}