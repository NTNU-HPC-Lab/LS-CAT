#ifndef HELPERS_H_
#define HELPERS_H_

#include <iostream>
#include <stdio.h>

#define ELECTRON_MASS 1.0f
#define PROTON_MASS 1836.2f
#define ELECTRON_CHARGE 1.0f
#define EPSILON_ZERO 1.0f
#define L 1.0f
#define pThreads 512
#define gThreadsSingle 10
#define gThreadsAll 1000
void CUDA_ERROR( cudaError_t err);

#endif
