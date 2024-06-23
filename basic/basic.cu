#include <iostream>
#include <memory>
#include <vector>

template<typename T>
std::shared_ptr<T> CreateTensor(size_t N)
{
    T * data;
    cudaMalloc(&data, N*sizeof(T));
    return std::shared_ptr<T>(data, [](T* data){ cudaFree(data);});
}

template<typename T, size_t N>
struct CudaVector
{
    std::array<T, N> host;
    void ToDevice()
    {
        deviceMemory = CreateTensor<T>(host.size());
        cudaMemcpy(deviceMemory.get(), host.data(), sizeof(T)*host.size(), cudaMemcpyHostToDevice);
    }

    void ToHost()
    {
        cudaMemcpy(host.data(), deviceMemory.get(), sizeof(T)*host.size(), cudaMemcpyDeviceToHost);
    }

    T * device_ptr() { return deviceMemory.get(); };


    std::shared_ptr<T> deviceMemory;
};

__global__ void VecAdd(float* x1, float* x2, float* y)
{
    int i  = threadIdx.x;
    y[i] = x1[i] + x2[i];
}

int main()
{
    CudaVector<float, 1024> x1;
    CudaVector<float, 1024> x2;
    CudaVector<float, 1024> y;

    for(int i = 0 ; i < x1.host.size(); ++i)
    {
        x1.host[i] = i*1.0f;
        x2.host[i] = 1024.0f - i*1.0f;
        y.host[i]  = 0.0f;
    }

    x1.ToDevice();
    x2.ToDevice();
    y.ToDevice();

    VecAdd<<<1, 1024>>>(x1.device_ptr(), x2.device_ptr(), y.device_ptr());
    y.ToHost();

    for(int i = 0 ; i < x1.host.size(); ++i)
    {
        assert(y.host[i] == x1.host[i] + x2.host[i] );
    }

    return 0;
}