#include "includes.h"
__global__ void kernelInterpolationCol(double *result, int rows, int cols, int factor){
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

// Puntos de referencia para interpolacion
double a,b;
double   m;

//
// Interpolacion de columnas
// -------------------------
while (x < cols*factor && y<rows){
int trueY = y*factor;
int offset = x + trueY*cols*factor;

a = result[ offset                     ];
b = result[ offset + cols*factor*factor];

m = (b - a)/((double)factor);

// Antes de llegar al final
if (y != rows-1){
for(int p=0; p<=factor; ++p){
result[offset] = a;
a += m;
offset += cols*factor*factor;
}
}

// Borde final
else{
for(int p=0; p<factor; ++p){
result[offset] = b;
b -= m;
offset += cols*factor*factor;
}
}
}

}