#include <iostream>
#include <memory>

template<typename T>
std::shared_ptr<T> CreateTensor(size_t N)
{
    T * data;
    cudaMalloc(&data, N*sizeof(T));
    return std::shared_ptr<T>(data, [](T* data){ cudaFree(data);});
}

__global__ void VecAdd(float* x1, float* x2, float* y)
{
    int i  = threadIdx.x;
    y[i] = x1[i] + x2[i];
}

int main()
{
    auto x1 = CreateTensor<float>(1024);
    auto x2 = CreateTensor<float>(1024);
    auto y  = CreateTensor<float>(1024);

    VecAdd<<<1, 1024>>>(x1.get(), x2.get(), y.get());

    
    return 0;
}