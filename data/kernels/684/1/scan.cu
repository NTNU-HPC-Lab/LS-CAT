#include "includes.h"



__global__ void scan(int *v, const int n)
{
int tIdx = threadIdx.x;
int step = 1;

while (step < n) {

int indiceDroite = tIdx;
int indiceGauche = indiceDroite + step;

if (indiceGauche < n) {
v[indiceDroite] = v[indiceDroite] + v[indiceGauche];
}

step = step * 2;
__syncthreads();

}

}