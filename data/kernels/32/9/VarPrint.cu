#include "includes.h"
__global__ void VarPrint(double *Var, int M, int N, int P){
for (int k=0; k < P; k++) {
for (int i=0; i < N; i++) {
for (int j = 0; j < M; j++) {
printf("%4.3f ", Var[k*M*N+i*M+j]);
}
printf("\n");
}
printf("\n"); printf("\n");
}
}