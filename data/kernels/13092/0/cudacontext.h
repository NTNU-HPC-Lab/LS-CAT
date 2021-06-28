//
// Created by deano on 26/04/16.
//
#pragma once
#ifndef VIZDOOM_CUDACONTEXT_H
#define VIZDOOM_CUDACONTEXT_H

#include "common.h"

class CudaContext {
public:
    typedef std::shared_ptr< CudaContext > ptr;

    CudaContext( int _gpuid );

    ~CudaContext();

    cudnnContext *getCudnnHandle() {
        return cudnnHandle;
    }

    cublasContext *getCublasHandle() {
        return cublasHandle;
    }

    int getGpuId() const {
        return gpuId;
    }

    // workspace system is a simple linear buffer of device memory for temporary calculation workspace
    void reserveWorkspace( const size_t size );

    void unreserveWorkspace( const size_t size );

    void *grabWorkspace( const size_t size );

    void releaseWorkspace( void *const ptr, const size_t size );

    size_t getWorkspaceSize() const {
        return maxWorkspaceRAM;
    }

protected:
    const int gpuId;
    size_t maxWorkspaceRAM;
    size_t curWorkspaceOffset;
    void *workspaceBase;


    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //VIZDOOM_CUDACONTEXT_H
