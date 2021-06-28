#include "includes.h"
__global__ void Print(float *beta, float *sigma, float *rho, int iter )
{
printf("\n %d -- 1) b %.5f -- s %.5f -- r %.5f ",iter,beta[0],sigma[0],rho[0]);
printf("\n %d -- 2) b %.5f -- s %.5f -- r %.5f ",iter,beta[1],sigma[1],rho[1]);
printf("\n %d -- 3) b %.5f -- s %.5f -- r %.5f ",iter,beta[2],sigma[2],rho[2]);


}