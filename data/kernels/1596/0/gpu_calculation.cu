#include "includes.h"

__global__ void gpu_calculation(float c0r, float c0i, float float_step, float imag_step, int *results, unsigned n, int W, int H, int inicial){

// index = m*x + y
const long unsigned globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

// printf("%d %d\n", blockIdx.x, threadIdx.x);

if (globalIndex < n) {
//calcular os complexos na mão
int x = (globalIndex + inicial)/W;
int y = (globalIndex + inicial)%H;
// printf("%d %d    %d\n", x, y, n);
float point_r = c0r+x*float_step;
float point_i = c0i+y*imag_step;

// printf("%f %f\n", point_r, point_i);
const int M = 1000;

// valor Zj que falhou
// -1 se não tiver falhado
int j = -1;

//Valor da iteração passada
float old_r = 0;
float old_i = 0;
float aux = 0;

//Calcula o mandebrot
for(int i = 1; i <= M; i++){

//Calculo da nova iteração na mão
aux = (old_r * old_r) - (old_i * old_i) + point_r;
old_i = (2 * old_r * old_i) + point_i;
old_r = aux;

//abs(complex) = sqrt(a*a + b*b)
//Passei a raiz do abs para outro lado
if( ((old_r * old_r + old_i * old_i) > 4 )){
j = i;
break;
}
}
// printf("%d\n", j);
// printf("%d\n", j);

results[globalIndex] = j;
// printf("%d\n", j);
}
// else printf("oh boy\n");

}