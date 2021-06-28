#include "includes.h"

#define TILE_SIZE 26
#define RADIUS 3
#define BLOCK_SIZE (TILE_SIZE+(2*RADIUS))

texture<unsigned char, 1, cudaReadModeElementType> texInImage;
texture<unsigned int, 1, cudaReadModeElementType> texIntegralImage;

__device__ unsigned int keypointsCount = 0;






__global__ void kernel_computeDesctriptorCUDARot(bool *_d_isdescriptor, char *_d_vdescriptor, int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation, int _amountofkeypoints, unsigned int *_d_integralImage, int _d_width, int _d_height, float _scale)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int index = bx + tx*_d_height;
_d_isdescriptor[index] = false;

if(index < _amountofkeypoints)
{
float S[64];
float _X[64];
float _Y[64];
float _Z[64];
float r, phi;
float pi = 3.1415926535f;

for(int i = 0 ; i < 64; i++)
{
r = _scale*pow(2.0f, 2+(i%4));
phi = (float)(i)/4.0f;
_X[i] = (r * cos ((2.0f * pi *phi)/16.0f));
_Y[i] = (r * sin ((2.0f * pi *phi)/16.0f));
_Z[i] = _scale * 8;
}

int _xIndex = _d_keypointsIndexX[index];
int _yIndex = _d_keypointsIndexY[index];
int tau = 4*_d_keypointsRotation[index];

bool check = true;
int index0;
int index1;
int index2;
int index3;

int _h_width = _d_width;
int _h_height = _d_height;

for(int i = 0 ; i < 64; i++)
{
if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width) < 0)check = false;
if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width) >= _h_width*_h_height)check = false;

if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width ) < 0)check = false;
if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width ) >= _h_width*_h_height)check = false;

if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width )< 0)check = false;
if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width )>= _h_width*_h_height)check = false;

if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width )< 0)check = false;
if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width )>= _h_width*_h_height)check = false;

if(check)
{
index0 = int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width);
index1 = int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width );
index2 = int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width );
index3 = int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width );

unsigned int a1 = tex1Dfetch(texIntegralImage, index0);
unsigned int a2 = tex1Dfetch(texIntegralImage, index1);
unsigned int a3 = tex1Dfetch(texIntegralImage, index2);
unsigned int a4 = tex1Dfetch(texIntegralImage, index3);

S[i] = float(a1+a2-a3-a4);
}
}

if(check)
{
_d_isdescriptor[index] = true;
bool desc[256];

for(int i = 0; i< 64; i++)
{
int id = (i+tau)%64;
int index0 = (id + 8)%64;
int index1 = (id + 24)%64;
int	index2 = (id + 36)%64;
int index3 = int((4.0f * id/4.0f  + 4.0f + (3 - (id%4))))%64;

if(S[id] < S[index0])
{
desc[i * 4] = true;
}else
{
desc[i * 4] = false;
}

if(S[id] < S[index1])
{
desc[i * 4 + 1] = true;
}else
{
desc[i * 4 + 1] = false;
}

if(S[id] < S[index2])
{
desc[i * 4 + 2] = true;
}else
{
desc[i * 4 + 2] = false;
}

if(S[id] < S[index3])
{
desc[i * 4 + 3] = true;
}else
{
desc[i * 4 + 3] = false;
}
}

for(int i = 0 ; i < 32; i++)
{
char wynik = 0;
for(int j = 0; j < 8; j++)
{
wynik += (desc[i * 8 + j] * (1 << j));
}
_d_vdescriptor[index*32 + i]=wynik;
}
}
}
}