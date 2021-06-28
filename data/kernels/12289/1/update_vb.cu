#include "includes.h"
__global__ void update_vb(float *d_verts_ptr, int vertex_count, float timeElapsed)
{
const unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;

if (threadId < vertex_count * 4)
{
float valx = d_verts_ptr[threadId * 4 + 0];
float valy = d_verts_ptr[threadId * 4 + 1];
float valz = d_verts_ptr[threadId * 4 + 2];


d_verts_ptr[threadId * 4 + 0] = valx * timeElapsed;
d_verts_ptr[threadId * 4 + 1] = valy * timeElapsed;
d_verts_ptr[threadId * 4 + 2] = valz * timeElapsed;
}
}