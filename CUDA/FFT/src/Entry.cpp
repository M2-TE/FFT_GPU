#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>

#include "Utils.cuh"
#include "KernelLauncher.cuh"
//#include "revised/fft_2w.cuh"
//#include "revised/fft_cg_2w.cuh"
#include "revised/fft_mk_cg_r2_2_1024.cuh"

void DoCUFFT();
int main()
{
	//DoA();
	ExecuteFFT<1024>();
	ExecuteFFT<1024>();
	//DoCUFFT();
}
void DoCUFFT()
{
#define NX 1024
#define BATCH 1

	cufftHandle plan;
	cufftComplex* data;
	float* input = new float[NX * 2];

	for (int i = 0; i < NX; i++)
	{
		input[2 * i] = i;
		input[2 * i + 1] = i;
	}

	cudaMalloc((void**)&data, sizeof(cufftComplex) * NX);
	cudaMemcpy(data, input, sizeof(cufftComplex) * NX, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	CUDA_TIMER_START();
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	//cufftExecC2C(plan, data, data, CUFFT_INVERSE);
	CUDA_TIMER_END();

	cudaDeviceSynchronize();

	printf("The  outputs are: \n");
	cudaMemcpy(input, data, sizeof(cufftComplex) * NX, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	//for (int l = 0; l < NX; l++) printf("RE:A[%d]=%.2f\t\t\t, IM: A[%d]=%.2f\t\t\t \n ", 2 * l, input[2 * l], 2 * l + 1, input[2 * l + 1]);
	CUDA_TIMER_PRINT();

	delete input;
	cufftDestroy(plan);
	cudaFree(data);
}