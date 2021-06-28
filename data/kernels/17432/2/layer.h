#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H

#define INPUT 0 
#define DENSE 1

#define SIGMOID 0

/* This is a header file for cudabrain, since it is compiled with nvcc, it has the CUDA extensions for C++ */ 

//class Layer
//{
//public:

/* Member Data */ 
  //  int type; // Dense, etc
    //int trainable; // 0 not trainable 1 trainable
   
    //float *d_data; //input data
    //int2 data_dims; //input data dimensions
    
    //float *d_offset;
    //float *d_weights;

    //float *d_output;
    //int units; //outputs of layer
//};
/*
class Input : public Layer
{
public:
    __host__
    Input(float *data, int2 in_data_dims)
    {
        type = INPUT;
        data_dims = in_data_dims;
        units = data_dims.y;

        cudaMalloc(&d_output, sizeof(data));
        cudaMemcpy(&d_output, data, sizeof(data), cudaMemcpyHostToDevice);
    }

    void dealloc(){
        cudaFree(d_output);
    }

};
*/

class Dense //: public Layer
{
public:
    int type;
    int trainable;

    float *d_data; //input data
    dim3 size; //input data dimensions
    
    float *d_bias;
    float *d_weights;
    int activation;

    float *d_output;
    int units; //outputs of layer

    float *deriv_error;

    /* Constructor: we create this class on the CPU and memory is copied to GPU for usage*/ 
    //__host__
public:
    Dense(float *d_data, int num_rows, int num_cols, 
          int units, int activation, int trainable)
    { 
        type = DENSE;

        this->d_data = d_data;
        size.x = num_rows;
        size.y = num_cols;
        size.z = units;
        this->units = units;
        this->activation = activation;
        this->trainable = trainable;
        cudaMalloc(&d_bias, units*sizeof(float));
        cudaMalloc(&d_weights, size.y*units*sizeof(float));
        cudaMalloc(&d_output, size.x*units*sizeof(float)); 
        cudaMalloc(&deriv_error, 1024 * units*sizeof(float));
    }

    void dealloc(){
        cudaFree(d_bias);
        cudaFree(d_weights);
        cudaFree(d_output);
        cudaFree(deriv_error);
    }
};

#endif
