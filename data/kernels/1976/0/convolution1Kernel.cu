#include "includes.h"


__shared__ int smem[324];

__global__ void convolution1Kernel(int *dst, int *src, int rows, int cols, int *filter) {
// Convolucion en memoria global, similar a la convolucion en CPU
int posx = threadIdx.x + blockIdx.x * blockDim.x;
int posy = threadIdx.y + blockIdx.y * blockDim.y;
if (posx > 0 && posy > 0 && posx < rows - 1 && posy < cols - 1) {
for (int k = 0; k < 3; ++k) {
for (int l = 0; l < 3; ++l) {
dst[posy * cols + posx] += src[(posy + k - 1) * cols + (posx + l - 1)] * filter[k * 3 + l];
//printf("Fuente = %i \n", src[(posy + k - 1) * cols + (posx + l - 1)]);
//printf("Filtro = %i \n", filter[k * 3 + l]);

}
}
}
//printf("Destino = %i \n", dst[posy * cols + posx]);

}