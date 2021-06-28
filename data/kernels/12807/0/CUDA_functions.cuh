#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>


__global__ void searchWithCuda(double *resultPossibilities, char *query, char *atribsValues, double *possibilities, int *queryPrefix, int *atribsPrefix, int *answersNumber, int *categoriesNumber, int *atribsNumber);

__host__ int findAnswer(char *query, char *atribsValues, double *possibilities, int *queryPrefix, int *atribsPrefix, int answersNumber, int categoriesNumber, int atribsNumber);