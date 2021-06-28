#include "includes.h"
// Gaurav Sheni
// CSC 391
// September 16, 2015
// Project 1



//declaring kernel call

__global__ void decrement(char* line, char* answer);

__global__ void decrement(char *current, char* answer){
int i = threadIdx.x;
answer[i] = (char)( (int) current[i] - 1 );
}