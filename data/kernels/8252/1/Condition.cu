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
__global__ void Condition(int *GrayScale_, int *d_changing1, int *cont, int linhas, int colunas, int flag)
{
int X_index[8]={-1,-1,0,1,1,1,0,-1};
int Y_index[8]={0,1,1,1,0,-1,-1,-1};
int neighbours[9]={0,0,0,0,0,0,0,0,0};
int i,j,total=0;
int ans=0;
int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
int fil = blockIdx.y * TILE_WIDTH + threadIdx.y;
int index = fil * colunas + col;
if (fil>0 && col>0 && fil < linhas-1 && col < colunas-1)
{
d_changing1[index]=0;
for(i=0; i<8; i++)
{
neighbours[i]=GrayScale_[(fil+X_index[i])*colunas + (col+Y_index[i])];
total+=neighbours[i];
}

neighbours[8]=total;
for(j=0; j<7; j++)
{
if(neighbours[j]==0 && neighbours[j+1]==1)
ans=ans+1;
}

if(neighbours[7]==0 && neighbours[0]==1)
ans=ans+1;
if(flag!=1)
{
if(GrayScale_[fil*colunas + col]==1 && neighbours[8]>=2 && neighbours[8]<=6
&& ans==1 && neighbours[0]*neighbours[2]*neighbours[4]==0
&& neighbours[2]*neighbours[4]*neighbours[6]==0)
{
d_changing1[index]=1;
cont[flag]=1;
}
}
else
{
if(GrayScale_[fil*colunas + col]==1 && neighbours[8]>=2 && neighbours[8]<=6
&& ans==1 && neighbours[0]*neighbours[2]*neighbours[6]==0
&& neighbours[0]*neighbours[4]*neighbours[6]==0)
{
d_changing1[index]=1;
cont[flag]=1;
}
}
}
}