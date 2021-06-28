//
// Author: Kaustubh Shivdikar
//


#ifndef FUNCTIONS_H
#define FUNCTIONS_H


#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include <errno.h>
#include <memory>
#include <stdexcept>

using namespace std;

typedef unsigned long long int uint64_cu;



// DEFINES

#define THREADS_PER_BLOCK 32


#define block_size   32
#define DEBUG 1
#define GPU 0
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define WARNINGS 1
#define WORD 32 // Size 



typedef struct PrimeHeader{
    uint64_cu lastMaxNo;
    uint64_cu length;
    uint64_cu* primelist;
}PrimeHeader;


typedef struct ThreadRetValue{
    uint64_cu length;
    uint64_cu* primelist;
}ThreadRetValue;


struct GpuHandler {
    int gpus=-1;
    uint64_cu* PL = NULL;
    uint64_cu PL_len = 0;
    uint64_cu IL_start = 1; 
    uint64_cu IL_end = 1; 
};



 long find_number_of_gpus();

void start_info();

void end_info();

void printList(uint64_cu* ilist, uint64_cu len);
ThreadRetValue* kernelLauncher(int gpu_id);

__global__ void prime_generator(int* d_input_list, uint64_cu* d_prime_list, uint64_cu* d_startPrimelist,uint64_cu* d_total_inputsize,uint64_cu* d_number_of_primes);

PrimeHeader calculate_primes_on_cpu(PrimeHeader pheader, uint64_cu pl_end_number);



 inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
 {
    if (code != cudaSuccess) 
    {
     fprintf(stderr,"\e[1;31mGPUassert: %s %s %d \e[0m\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
 }


#define INTSIZE sizeof(uint64_cu)
void writePrimes(uint64_cu* primes, uint64_cu length, uint64_cu lastNo);
PrimeHeader readPrimes();

#endif // FUNCTIONS_H
