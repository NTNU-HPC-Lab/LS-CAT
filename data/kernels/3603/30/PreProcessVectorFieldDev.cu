#include "includes.h"
__global__	void	PreProcessVectorFieldDev(float3*	field, float width, float height, float minx, float miny)
{
uint	idx = threadIdx.x*gridDim.x + blockIdx.x;

float3	p1 = field[idx*2+0];
float3	p2 = field[idx*2+1];

p1.x -= minx;
p1.y -= miny;
p1.z = 0;

p1.x /= width;
p1.y /= height;

p2.x -= minx;
p2.y -= miny;
p2.z = 0;

p2.x /= width;
p2.y /= height;

field[idx*2+0] = p1;
field[idx*2+1] = p2;
}