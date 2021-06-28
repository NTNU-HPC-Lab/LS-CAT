#ifndef PYRDOWN_H
#define PYRDOWN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#include "global_var.h"


class PyrDown_gpu
{

public:


    u_int8_t *ptImageTmp_Device;
    u_int8_t *ptImageL0_Device;
    u_int8_t *ptImageL1_Device;
    u_int8_t *ptImageL2_Device;
    u_int8_t *ptImageL3_Device;

    PyrDown_gpu(int rows, int cols);
    ~PyrDown_gpu();
    void run(int rows,int cols,u_int8_t *ptSrc);
    static void run(int rows,int cols,u_int8_t *ptIn,u_int8_t *ptOut,u_int8_t * ptTmp);
};

#endif // PYRDOWN_H
