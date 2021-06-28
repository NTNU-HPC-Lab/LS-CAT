#include "includes.h"
__global__ void MatrixMulKernel(int * _matrixA, int * _matrixB, int * _result, int _width)
{
int k = 0, elementA = 0, elementB = 0;
//2D thread ID
int tx = threadIdx.x;
int ty = threadIdx.y;

//valeu store the _result element that is computed by thread
int value = 0;
for (k = 0; k < _width; k++)
{
elementA = *(_matrixA + (ty*_width + k));  //Go accross the line
elementB = *(_matrixB + (k*_width + tx));  //Go accross the column
value += (elementA * elementB);   //Take each element
}
*(_result + (_width*ty + tx)) = value;

return;
}