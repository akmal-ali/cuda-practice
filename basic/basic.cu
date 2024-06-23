#include <iostream>


__global__ void VecAdd(float* x1, float* x2, float* y)
{
    int i  = threadIdx.x;
    y[i] = x1[i] + x2[i];
}

void main()
{
    float* x1 = cudaMalloc(1024*sizeof(float));
    float* x2 = cudaMalloc(1024*sizeof(float));
    float* y  = cudaMalloc(1024*sizeof(float));

    VecAdd<<<1, 1024>(x1, x2, y);

    // Free memory.
    cudaFree(x1); x1 = nullptr;
    cudaFree(x2); x2 = nullptr;
    cudaFree(y);   y = nullptr;
}