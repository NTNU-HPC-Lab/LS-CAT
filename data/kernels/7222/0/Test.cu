#include "includes.h"

#define INPUTS 5
#define H_LAYERS 3
#define H_HEIGHT 5
#define OUTPUTS 1
#define BIAS 0
#define alpha_d 0.5

#define DATA_FILE "data_for_training.txt"
#define TEST_FILE "data_for_verify.txt"

#define ITERATIONS 10000

//#define DEBUG

//Weight declarations
double weights_in[INPUTS * H_HEIGHT];// = {.15,.20};
double weights_out[OUTPUTS * H_HEIGHT];// = {.65,.7};
__global__ void Test(double *training_in_d, double *training_out_d, double *data_range_d, double *weights_in_d, double *weights_out_d, double *weights_h_d, double *h_out_d, double *outputs_d, int inputs, int samples, int height){
printf("\nDevice:\n");

printf("Training In: %f\n", training_in_d[40 * inputs + 3]);
printf("Training Out: %f\n", training_out_d[40]);
printf("Data Range: %f\n", data_range_d[5 * 2 + 1]);
printf("Weights In: %f\n", weights_in_d[325]);
printf("Weights Out: %f\n", weights_out_d[50]);
printf("Weights H: %f\n", weights_h_d[5 * height * height + 50]);
printf("H Out: %f\n", h_out_d[5 * height + 50]);
printf("Outputs: %f\n", outputs_d[0]);
}