// Google colab - pick a machine with an Nvidia GPU!
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include "common.cuh"

// Vector add!
__global__ void VecAdd(float* x1, float* x2, float* y, int N)
{
    int i  = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        y[i] = x1[i] + x2[i];
}

int main()
{
    printf("Starting %s \n", __FILE__);
    cudaFree(0);
    const size_t N = 1920*1080;

    CudaVector<float, N> x1;
    CudaVector<float, N> x2;
    CudaVector<float, N> y;

    for(int i = 0 ; i < x1.host.size(); ++i)
    {
        x1.host[i] = -1.0*i;
        x2.host[i] = 2*i;
        y.host[i]  = 0.0f;
    }

    x1.ToDevice();
    x2.ToDevice();
    y.ToDevice();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
    // Run kernel
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(
        x1.device_ptr(), x2.device_ptr(), y.device_ptr(),
        N);

    CUDA_CHECK(cudaDeviceSynchronize());
    y.ToHost();

    for(int i = 0 ; i < x1.host.size(); ++i)
    {
        assert(x1.host[i] + x2.host[i] == y.host[i]);
    }

    printf("Hello world\n");
    std::cout << " finished successfully" << std::endl;
    return 0;
}