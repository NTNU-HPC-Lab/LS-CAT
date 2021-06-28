#include "includes.h"


// includes, project

#define PI 3.1415926536f


int MaxThreadsPerBlock;
int MaxThreadsX;
int MaxThreadsY;


// Conversion d'un vecteur réel en vecteur complexe

// Conversion d'un vecteur complexe en vecteur réel


// Multiplie point par point un vecteur complex par un vecteur réel

// Applique y = at*x +bt à chaque point d'un vecteur réel




// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de réel
// Alpha n'est pas modifié

// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de bytes
// Alpha n'est pas modifié


// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de réel
// Alpha autorise l'affichage au dessus d'un certain seuil


// Processus auto-régressif X2 = a*X1 + b*X0 + N0;



// Expansion
// On applique une interpolation bi-linéaire à la source

// Transformation Cartesian To Polar
// On applique une interpolation bi-linéaire à la source



__global__ void FillTex(void *surface, int width, int height, size_t pitch, double* src, int Mask)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

unsigned char *pixel1;

if (x >= width || y >= height) return;

double w = src[x + width*y];

if (w<0) {w=0;}
if (w>253) {w=253;}


pixel1 = (unsigned char *)( (char*)surface + y*pitch) + 4*x;

//pixel1[3] = 255;                     // alpha = 255 sauf s'il fait partie du masque
for (int i=0;i<4;i++)
{  if (Mask & (1<<i))  pixel1[i] = w;  }
}