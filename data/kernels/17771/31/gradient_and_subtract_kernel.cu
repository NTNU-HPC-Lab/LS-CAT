#include "includes.h"
__global__ void gradient_and_subtract_kernel(float * in, float * grad_x, float * grad_y, float * grad_z)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
return;

long int id = (k * c_Size.y + j) * c_Size.x + i;
long int id_x = (k * c_Size.y + j) * c_Size.x + i + 1;
long int id_y = (k * c_Size.y + j + 1) * c_Size.x + i;
long int id_z = ((k + 1) * c_Size.y + j) * c_Size.x + i;

if (i != (c_Size.x - 1))
grad_x[id] -= ((in[id_x] - in[id]) / c_Spacing.x);
if (j != (c_Size.y - 1))
grad_y[id] -= ((in[id_y] - in[id]) / c_Spacing.y);
if (k != (c_Size.z - 1))
grad_z[id] -= ((in[id_z] - in[id]) / c_Spacing.z);
}