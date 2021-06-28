//
// Created by yanhao on 17-11-20.
//

#ifndef CUDA_LEARN_MATRIXMUL_H
#define CUDA_LEARN_MATRIXMUL_H

#include "utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h> //cuda自带库函数

//用于指示不同的GPU 优化版本
enum Type {
    Mode1 = 1,   //Mode 1 :将每一个C[i][j]都分别分配一个线程
    Mode2 = 2,     //Mode 2 :不让一个线程完整计算一个C[i][j]，通过C(i,j) = sum { A(i,k)*B(k,j) }发现，我们还可以再细度划分：
    //           sub(i,j) = sum{A(i,ksub+offsetA)*B(ksub+offsetB,j)}  0<=ksub < blockSize
    //            C(i, j) = sum{ Csub(i, j) }
    //            就是把矩阵分成n*n个大的子块，然后每一个block负责计算子块i 和 子块j的子乘积，计算完毕后加起来则可。这里主要使用了共享显存作优化。
    Mode3=3
};
extern "C" {
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                        unsigned int HB, Type mode);
cublasStatus_t addWithCuda2(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                           unsigned int HB);

}


#endif //CUDA_LEARN_MATRIXMUL_H
