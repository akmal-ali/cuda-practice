#pragma once

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
