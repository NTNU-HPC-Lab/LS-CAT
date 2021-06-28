#include "includes.h"
__global__ void colorVectorField( float3 *colors, float3 *colorMap, float2 *__restrict__ field, dim3 blocks, unsigned int simWidth, unsigned int simHeight)
{
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

int quadIdx = x + simWidth*y;
if (x == simWidth/2 + 10 && y == simWidth/2 + 10)
printf("coloring vector field values on the order of: %f, %f\n", field[quadIdx].x, field[quadIdx].y);

//float mag = sqrt(field[quadIdx].x * field[quadIdx].x + field[quadIdx].y * field[quadIdx].y);
float mag = field[quadIdx].x;
int map = (int)(mag/0.004* 256);
if(map > 255) { map = 255; }
if(map < 0) { map = 0; }

for(int i = 0; i < 4; i++){
colors[4*quadIdx+i] = colorMap[map];
if (field[quadIdx].x == 0)
{
colors[4*quadIdx+i] = make_float3(0.0, 0.6, 0.2);
}
if (mag < 0)
{
colors[4*quadIdx+i] = make_float3(0.0, 0.3, 0.7);
}
}

/*
for(int i = 0; i < 4; i++){
if (newVel[quadIdx].x < 0){
colors[4*quadIdx+i].x = 1.0;
}
}*/
}