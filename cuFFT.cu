#include <iostream> 
#include <cufft.h> 
#include <cuda_runtime.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#pragma comment(lib, "D:/Cuda_Juliya/lib/glut64.lib")
// "D:/Cuda_Juliya/bin/glut64.dll"
#include "D:/Cuda_Juliya/common/book.h"
#include "D:/Cuda_Juliya/common/cpu_bitmap.h"
#include <stdio.h>
#include <time.h>

#define N 32768 // Size of the FFT, Must be a power of 2 


int main() 
{ 
	cufftHandle plan;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float sumtime = 0;
	for (int j = 0; j < 10; ++j)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cufftComplex h_input[N]; // Input array (complex numbers) 
		cufftComplex h_output[N]; // Output array (complex numbers) 

		for (int i = 0; i < N; ++i)
		{
			h_input[i].x = i; // Real part 
			h_input[i].y = 0; // Imaginary part 
		}

		cufftComplex* d_input, * d_output;
		cudaMalloc((void**)&d_input, sizeof(cufftComplex) * N);
		cudaMalloc((void**)&d_output, sizeof(cufftComplex) * N);
    
		cudaMemcpy(d_input, h_input, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);
    
		cufftPlan1d(&plan, N, CUFFT_C2C, 1); // CUFFT_C2C for complex-to-complex transform 

		cudaEventRecord(start, 0);

		cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
		
    cudaEventRecord(stop, 0);
		
    cudaDeviceSynchronize();
		
    cudaMemcpy(h_output, d_output, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

		cudaEventSynchronize(stop);
		
    float elapsedTime;
		
    cudaEventElapsedTime(&elapsedTime, start, stop); 
		
    sumtime += elapsedTime;
		
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cufftDestroy(plan);
		cudaFree(d_input);
		cudaFree(d_output);
		std::cout << "FFT Output:" << std::endl;
		for (int i = 0; i < 8; ++i)
		{
			std::cout <<  " (" << h_output[i].x << ", " << h_output[i].y << ")" << std::endl;
		}
	}
	printf("Time taken: %f s\n", sumtime/10);
	return 0; 
} 
