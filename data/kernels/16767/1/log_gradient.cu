#include "includes.h"

//FILE IO RELATED
//max number of lines in the training dataset
#define MAX_ROWS_TRAINING 16896
// max number of columns/features in the training dataset
#define MAX_COLUMNS_TRAINING 26
// max number of rows in the testing dataset
#define MAX_ROWS_TESTING 4096
// max number of columns in the testing data
#define MAX_COLUMNS_TESTING 26
//max number of characters/line
#define MAX_CHAR 300

__constant__ int features = 26;
__constant__ int num_rows = 16896;

long mem_cpy_time = 0;
long beta_cpy_time = 0;

// parallelized across the rows

// parallelized across the features

__global__ void log_gradient(float* log_func_v,  float* gradient, float* betas, float* data, int* yvec) {
// the logistic function itself has been pulled out
int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
float temp = 0.0f;
for(int i = 0; i < num_rows; i++) {
float sub = log_func_v[i] - yvec[i];
float accessed_data = data[(i * features) + feature_index];
temp += sub * accessed_data;
}
gradient[feature_index] = temp;
}