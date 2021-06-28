#include "includes.h"

#ifdef TIME
#define COMM 1
#elif NOTIME
#define COMM 0
#endif

#define MASK_WIDTH 5
#define TILE_WIDTH 32
#define GPU 1
#define COMMENT "skeletization_GPU"
#define RGB_COMPONENT_COLOR 255


typedef struct {
unsigned char red, green, blue;
} PPMPixel;

typedef struct {
int x, y;
PPMPixel *data;
} PPMImage;

typedef struct {
int x, y;
} Par;

double time_total;
__global__ void Update(int *GrayScale_, int *d_changing1, int linhas, int colunas)
{
int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
int fil = blockIdx.y * TILE_WIDTH + threadIdx.y;
int index = fil * colunas + col;

if (fil>0 && col>0 && fil < linhas-1 && col < colunas-1)
{
if(d_changing1[index]==1)
GrayScale_[index]=0;
}
}