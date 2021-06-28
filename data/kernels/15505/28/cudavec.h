#pragma once

#include "util.h"
#include <cuda.h>
#include <vector>

#include <iostream>

extern uint64_t memory_counter;

namespace dexe {


template <typename F> struct CudaVec {
    F *data = 0;
    int N = 0;
    bool own = true;

    CudaVec() : data(0), N(0) {}
    CudaVec(int n_) : data(0), N(0) { allocate(n_); }
    CudaVec(F *data, int n_) : data(data), N(n_), own(false) {}

    ~CudaVec() {
        if (own && N) {
            memory_counter -= N;
            handle_error(cudaFree(data));
        }
    }

    CudaVec &operator=(CudaVec &other) {
        if (N != other.N) {
            allocate(other.N);
        }
        copy_gpu_to_gpu(other.data, data, N);
        return *this;
    }

    void allocate(int n);

    bool allocated() {
        return data != NULL;
    }

    void free() {
        allocate(0);
    }

    CudaVec(CudaVec &other) {
        allocate(other.N);
        copy_gpu_to_gpu(other.data, data, N);
    }

    void zero(int offset = 0) {
        if (N)
            handle_error(cudaMemset(data + offset, 0, sizeof(F) * (N - offset)));
    }

    void init_normal(F mean, F std) { dexe::init_normal<F>(data, N, mean, std); }

    void add_normal(F mean, F std) { dexe::add_normal<F>(data, N, mean, std); }

    void share(CudaVec<F> &other) {
        if (N != other.N) {
            throw DexeException("can't share with CudaVec of different size");
        }
        if (own)
            cudaFree(data);
        own = false;
        data = other.data;
    }

    std::vector<F> to_vector() {
        std::vector<F> vec(N);
        handle_error(cudaMemcpy(&vec[0], data, N * sizeof(F), cudaMemcpyDeviceToHost));
        return vec;
    }

    void to_ptr(F *target) {
        handle_error(cudaMemcpy(target, data, N * sizeof(F), cudaMemcpyDeviceToHost));
    }

    void from_ptr(F const *source) {
        handle_error(cudaMemcpy(data, source, N * sizeof(F), cudaMemcpyHostToDevice));
    }

    void from_vector(std::vector<F> const &vec) {
        if (vec.size() != N)
            allocate(vec.size());
        handle_error(cudaMemcpy(data, &vec[0], N * sizeof(F), cudaMemcpyHostToDevice));
    }

    F sum() {
        F result(0);
        if (sizeof(F) == sizeof(float))
            handle_error(cublasSasum(Handler::cublas(), N, data, 1, &result));
        else
            handle_error(cublasDasum(Handler::cublas(), N, data, 1, &result));
        return result;
    }

    CudaVec<F> &sqrt();
    CudaVec<F> &abs();
    CudaVec<F> &pow(F e);
    CudaVec<F> &exp();
    CudaVec<F> &clip(F limit);
    CudaVec<F> &add(int idx, F val);

    CudaVec<F> &operator-=(CudaVec<F> &other);
    CudaVec<F> &operator+=(CudaVec<F> &other);
    CudaVec<F> &operator*=(CudaVec<F> &other);
    CudaVec<F> &operator/=(CudaVec<F> &other);
    CudaVec<F> &operator/=(F val);

    CudaVec &operator*=(F v);
    CudaVec &operator+=(F v);

    // Remove copy operator to be safe
    CudaVec<F>(const CudaVec<F> &) = delete;
};

__global__ void sqrt_kernel(float *v, int n);
__global__ void pow_kernel(float *v, int n, float e);
__global__ void times_kernel(float *v, float *other, int n);
__global__ void add_scalar(float *v, float other, int n);
__global__ void times_scalar(float *v, float other, int n);
__global__ void divide_kernel(float *v, float *other, int n);
__global__ void abs_kernel(float *v, int n);
__global__ void exp_kernel(float *v, int n);
__global__ void clip_kernel(float *v, int n, float limit);

__global__ void sqrt_kerneld(double *v, int n);
__global__ void pow_kerneld(double *v, int n, double exp);
__global__ void times_kerneld(double *v, double *other, int n);
__global__ void add_scalard(double *v, double other, int n);
__global__ void times_scalard(double *v, double other, int n);
__global__ void divide_kerneld(double *v, double *other, int n);
__global__ void abs_kerneld(double *v, int n);
__global__ void exp_kerneld(double *v, int n);
__global__ void clip_kerneld(double *v, int n, double limit);

} // namespace dexe
