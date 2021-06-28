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



__global__ void Kernel_CartToPol1(double *tb1, double *tb2, int width, int height )
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x >= width || y >= height) return;

double R = 2* sqrtf( powf(x-width/2,2) + powf(y-height/2,2) );
double theta = (atan2f( y-height/2 ,  x-width/2) +PI)*height/(2*PI);

int x1 = ((int) R) % width ;
int y1 = ((int) theta) % height;

int xp1 = (x1+1) % width;
int yp1 = (y1+1) % height;

double z1 =   tb1[width*y1+x1];
double z2 =   tb1[width*yp1+x1];
double z3 =   tb1[width*yp1+xp1];
double z4 =   tb1[width*y1+xp1];

double dx =  theta-floorf(theta);
double dy =  R-floorf(R);

double zp = 1.0*z1+ dy*(1.0*z2-z1);
double zq = 1.0*z4+ dy*(1.0*z3-z4);
double ZR = zp+ dx*(zq-zp);

tb2[width*y+x] = ZR;

}