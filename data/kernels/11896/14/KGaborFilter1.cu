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



__global__ void KGaborFilter1(double* filter, double* Vr, int width, int height, double ss , double r0, double sr0, double stheta0 )
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if (i >= width || j >= height) return;

double x = i;
double y = j;

if (i> width/2)  x = width-i;
if (j> height/2) y = height-j;

#define Eps 1E-6;
double r = sqrt(x*x+ y*y)+Eps;

double theta;
if (x>0) theta= atan2( y, x); else theta = PI/2;

//double ff =  exp( cos(2*theta)/stheta0 )
//             *
//             exp(-0.5*pow(log(r/r0),2)/log(1+pow(sr0,2))) * pow(r0/r,3)*ss*r;

// Correction Jonathan 7-12-16
double ff =  exp( cos(2*theta)/(4*pow(stheta0,2) ) )
*
exp(-0.5*pow(log(r/r0),2)/log(1+pow(sr0,2))) * pow(r0/r,3)*4*pow(ss*r,3);


filter[i+j*width] = ff;
if (i>0 || j>0) Vr[i+j*width] =  ff/(4*pow(ss*r,3)); else Vr[i+j*width] = 0;


}