
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#pragma comment(lib, "D:/Cuda_Juliya/lib/glut64.lib")
// "D:/Cuda_Juliya/bin/glut64.dll"
#include "D:/Cuda_Juliya/common/book.h"
#include "D:/Cuda_Juliya/common/cpu_bitmap.h"

#include <stdio.h>

#define DIM 10

__global__ void matrixkernel(double* c, const double* a, const double* b)
{
    double sum = 0;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = y * gridDim.x;
    for (int i = 0; i < gridDim.x; ++i) {
        sum = sum + a[offset + i] * b[x * gridDim.x + i];
    }
    c[x + offset] = sum;
}

void print_matrix(double* a)
{
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            printf("%lg ", a[j + DIM * i]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_input(double* a)
{
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            a[j + DIM * i] = j * i + i;
        }
    }
}

void matix_transpose(double* a)
{
    double temp;
    for (int i = 0; i < DIM; ++i) {
        for (int j = i + 1; j < DIM; ++j) {
            temp = a[j + DIM * i];
            a[j + DIM * i] = a[i + DIM * j];
            a[i + DIM * j] = temp;
        }
    }
}

int matrix_multiplication()
{
    cudaError_t cudaStatus;
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    double a[DIM * DIM] = { 0 };
    double b[DIM * DIM] = { 0 };
    double c[DIM * DIM] = { 0 };

    matrix_input(a);
    matrix_input(b);

    print_matrix(a);
    print_matrix(b);
    matix_transpose(b);

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, DIM * DIM * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, DIM * DIM * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, DIM * DIM * sizeof(double)));

    cudaStatus = cudaMemcpy(dev_a, a, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    dim3    grid(DIM, DIM);
    matrixkernel << <grid, 1 >> > (dev_c, dev_a, dev_b);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, DIM * DIM * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }


    print_matrix(c);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    //cudaDeviceReset must be called before exiting in order for profiling and
//tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


int main()
{
    int error = matrix_multiplication();  
    return error;
}

