#include "includes.h"
__global__ void kernelInterpolationRow(double *original, double *result, int rows, int cols, int factor){
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

int idOriginal,idResult;

// Puntos de referencia para interpolacion
double a,b;
double   m;

//
// Interpolacion de filas
// ----------------------
while (x < rows){
idOriginal = y*rows               + x       ;
idResult   = y*rows*factor*factor + x*factor;

a = original[ idOriginal    ];
b = original[ idOriginal + 1];

m = (b - a)/((double)factor);

// Antes de llegar al final
if (x != rows-1){
for(int p=0; p<=factor; ++p){
result[idResult] = a;
a += m;
++idResult;
}
}

// Borde final
else{
for(int p=0; p<factor; ++p){
result[idResult] = b;
b -= m;
++idResult;
}
}

}

}