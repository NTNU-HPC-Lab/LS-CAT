#include "MyClass.h"

class MyCudaClass : public MyClass {
    public:
        using Base = MyClass;

        double *devParam;
        const size_t dubSize = sizeof(double);

        int nCudaBlocks = 1;
        int nCudaThreadsPerBlock = 1;

        MyCudaClass(){
            cudaMalloc((void **)&devParam, dubSize);
        }

        ~MyCudaClass(){
            cudaFree(devParam);
        }

        void set_param(double in){
            Base::set_param(in);
            cudaMemcpy(devParam, &(Base::hostParam), dubSize, cudaMemcpyHostToDevice);
        }

        double do_it_on_device(){
            double *devOut, out;
            cudaMalloc((void **)&devOut, dubSize);
            devKernel<<< nCudaBlocks, nCudaThreadsPerBlock >>>(devParam, devOut);
            cudaMemcpy(&out, devOut, dubSize, cudaMemcpyDeviceToHost);
            return out;
        }

        __global__ static void devKernel(double *param, double *ans){
            // Cuda implementation
            std::printf("Inside devKernel: ");
            *ans = *param + 3.14;
        }

};