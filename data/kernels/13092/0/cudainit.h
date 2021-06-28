//
// Created by deano on 26/04/16.
//
#pragma once
#ifndef VIZDOOM_CUDAINIT_H
#define VIZDOOM_CUDAINIT_H

#include <memory>
#include <sstream>
#include <iostream>

int cudaInit();

void cudaShutdown();

int cudaGetContextCount();

std::shared_ptr< class CudaContext > cudaGetContext( int gpuId );

#endif //VIZDOOM_CUDAINIT_H
