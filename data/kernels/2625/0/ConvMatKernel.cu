#include "includes.h"

typedef struct bmpFileHeaderStruct {
/* 2 bytes de identificación */
uint32_t size;        /* Tamaño del archivo */
uint16_t resv1;       /* Reservado */
uint16_t resv2;       /* Reservado */
uint32_t offset;      /* Offset hasta hasta los datos de imagen */
} bmpFileHeader;

typedef struct bmpInfoHeaderStruct {
uint32_t headersize;  /* Tamaño de la cabecera */
uint32_t width;       /* Ancho */
uint32_t height;      /* Alto */
uint16_t planes;      /* Planos de color (Siempre 1) */
uint16_t bpp;         /* bits por pixel */
uint32_t compress;    /* compresion */
uint32_t imgsize;     /* tamaño de los datos de imagen */
uint32_t bpmx;        /* Resolucion X en bits por metro */
uint32_t bpmy;        /* Resolucion Y en bits por metro */
uint32_t colors;      /* colors used en la paleta */
uint32_t imxtcolors;  /* Colores importantes. 0 si son todos */
} bmpInfoHeader;


__global__ void ConvMatKernel(unsigned char *img_device, unsigned char *img_device2, uint32_t width_image, uint32_t height_image, int j, float *mat) {
//Hay que pasarle la matriz
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int i = width_image * row + col;
float avgB, avgG, avgR;
int x, y;

avgB = avgG = avgR = 0;

if (i < (width_image * height_image)) {
for(x = -1; x < 2; x++) {
if (row == 0 && x == -1) {
x = 0;
}
else if (row == height_image - 1) {
if (x > 0) break;
}
for(y = -1; y < 2; y++) {
if (col == 0 && y == -1) y = 0;
if (col == width_image - 1 && y == 1) break;
avgB += img_device[(col + y)*3 + (x + row) * width_image*3 + 0] * mat[((x + 1) * 3) + y + 1];
avgG += img_device[(col + y)*3 + (x + row) * width_image*3 + 1] * mat[((x + 1) * 3) + y + 1];
avgR += img_device[(col + y)*3 + (x + row) * width_image*3 + 2] * mat[((x + 1) * 3) + y + 1];
}
}
img_device2[col*3 + row*width_image*3 + 0] = avgB;
img_device2[col*3 + row*width_image*3 + 1] = avgG;
img_device2[col*3 + row*width_image*3 + 2] = avgR;
}
}