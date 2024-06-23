#include <iostream>
#include <memory>
#include <vector>
#include <cassert>


// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}


template<typename T>
std::shared_ptr<T> CreateTensor(size_t N)
{
    T * data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, N*sizeof(T)));
    return std::shared_ptr<T>(data, [](T* data){ 
        CUDA_CHECK(cudaFree(data));
    });
}

template<typename T, size_t N>
struct CudaVector
{
    std::array<T, N> host;
    void ToDevice()
    {
        mDevice = CreateTensor<T>(N);
        CUDA_CHECK(cudaMemcpy(mDevice.get(), host.data(), sizeof(T)*N, cudaMemcpyHostToDevice));
    }

    void ToHost()
    {
        CUDA_CHECK(cudaMemcpy(host.data(), mDevice.get(), sizeof(T)*N, cudaMemcpyDeviceToHost));
    }

    T * device_ptr() { return mDevice.get(); };


    std::shared_ptr<T> mDevice;
};

__global__ void VecAdd(float* x1, float* x2, float* y)
{
    int i  = threadIdx.x;
    y[i] = 100.0f;//x1[i] + x2[i];
}

int main()
{
    cudaFree(0);
    cuda_hello<<<1,1>>>(); 


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
    CUDA_CHECK(cudaDeviceSynchronize());
    y.ToHost();

    for(int i = 0 ; i < x1.host.size(); ++i)
    {
        std::cout << x1.host[i] << " + " << x2.host[i] << " =" << y.host[i] << std::endl;
    }

    return 0;
}