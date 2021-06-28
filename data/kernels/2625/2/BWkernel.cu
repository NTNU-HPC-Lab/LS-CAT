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


__global__ void BWkernel(unsigned char *img_device, uint32_t n) {
float color;
color = 0.0f;
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
color += img_device[i*3 + 0] * 0.114;
color += img_device[i*3 + 1] * 0.587;
color += img_device[i*3 + 2] * 0.299;
color /= 3;
img_device[i*3 + 0] = color;
img_device[i*3 + 1] = color;
img_device[i*3 + 2] = color;
}
}