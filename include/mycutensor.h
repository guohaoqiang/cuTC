#ifndef MYCUTENSOR_H
#define MYCUTENSOR_H 
#include "tensor.h"

#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

void run_cuTensor(Tensor& A, Tensor& B, Tensor& C);

#endif /* MYCUTENSOR_H */
