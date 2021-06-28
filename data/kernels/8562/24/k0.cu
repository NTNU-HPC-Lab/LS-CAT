#include "includes.h"
__global__ void k0( float* g_dataA, float* g_dataB, int pitch, int width )
{

// global thread(data) row index
unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
i = i + 1; //because the edge of the data is not processed

// global thread(data) column index
unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
j = j + 1; //because the edge of the data is not processed

// check the boundary
if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;

g_dataB[i * pitch + j] = (
0.2f * g_dataA[i * pitch + j] +               //itself
0.1f * g_dataA[(i-1) * pitch +  j   ] +       //N
0.1f * g_dataA[(i-1) * pitch + (j+1)] +       //NE
0.1f * g_dataA[ i    * pitch + (j+1)] +       //E
0.1f * g_dataA[(i+1) * pitch + (j+1)] +       //SE
0.1f * g_dataA[(i+1) * pitch +  j   ] +       //S
0.1f * g_dataA[(i+1) * pitch + (j-1)] +       //SW
0.1f * g_dataA[ i    * pitch + (j-1)] +       //W
0.1f * g_dataA[(i-1) * pitch + (j-1)]         //NW
) * 0.95f;
}