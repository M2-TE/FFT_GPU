#define N 256
#define M 2*N
#define D 2*N
//n is the number of stages in 256-point, radix-2² FFT(log256/log2).
#define n 8

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

void printDeviceInfo();

// DEVICE (thread)
__global__ void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SROT[M];

	short j = threadIdx.x;
	//loading "rotation elements" from Global memory to Shared memory with coalescence(in order of thread indices).
	SROT[j] = ROT[j];
	//loading "rotation elements" from Global memory to Shared memory with coalescence(in order of thread indices).
	//SA[j] = A[j];
	//synchronize all the threads untill all threads done their work(loading).
	//__syncthreads();
	short i = j >> 1;
	short k = j % 2;
	short l = i >> 1;
	short m = i % 2;
	//r0 computes the rotation indeces, which is the same in all odd stages.
	//r1,r3,r5 computes the rotation indeces for even stages, which differs from one to another.
	short r0 = m * (i >> (n - 1));
	short signk = -(k << 1) + 1;
	short signm = -(m << 1) + 1;
	short x = r0 * (signk);
	short signkr0 = -2 * ((!k) * r0) + 1;
	short ind0 = l << 1;
	short ind1 = (l << 1) + N;
	short s;


	//FFT computations will be done from stage1 to stage8 as below.
	//stage1:
	SB[j + x] = signkr0 * (A[ind0 + k] + signm * A[ind1 + k]);
	__syncthreads();

	//stage2:
	s = 2;
	short r1 = (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SA[j] = SB[ind0 + k] + signm * SB[ind1 + k];
	//__syncthreads();
	SB[j] = signk * SA[i << 1] * SROT[(r1 << 1) + k] + SA[(i << 1) + 1] * SROT[(r1 << 1) + (!k)];
	__syncthreads();

	//stage3:
	SA[j + x] = signkr0 * (SB[ind0 + k] + signm * SB[ind1 + k]);
	__syncthreads();

	//stage4:
	s = 4;
	short r3 = 4 * (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SB[j] = SA[ind0 + k] + signm * SA[ind1 + k];
	//__syncthreads();
	SA[j] = signk * SB[i << 1] * SROT[(r3 << 1) + k] + SB[(i << 1) + 1] * SROT[(r3 << 1) + (!k)];
	__syncthreads();

	//stage5:
	//SB[j+x] = signkr0*(SA[ind0+k] + (signm)*SA[ind1+k]);
	SB[j + x] = signkr0 * (SA[ind0 + k] + signm * (SA[ind1 + k]));
	//__syncthreads();
	
	//stage6:
	s = 6;
	short r5 = 16 * (((i % 2) << 1) + ((i >> 1) % 2)) * (i >> s);
	SA[j] = SB[ind0 + k] + signm * SB[ind1 + k];
	//__syncthreads();
	SB[j] = signk * SA[i << 1] * SROT[(r5 << 1) + k] + SA[(i << 1) + 1] * SROT[(r5 << 1) + (!k)];
	__syncthreads();

	//stage7:
	SA[j + x] = signkr0 * (SB[ind0 + k] + signm * (SB[ind1 + k]));
	//__syncthreads();
	
	//stage8:
	A[j] = SA[ind0 + k] + signm * SA[ind1 + k];
	//synchronize all the threads untill all threads done their work(storing).
	//__syncthreads();
	//storing output elements from Shared memory to Global memory with coalescence(in order of thread indices).
	//A[j] = SB[j];
}

// HOST
int main()
{
	size_t nBytes = 2 * N * sizeof(float);
	float A[2 * N]; // both input and output
	float Rot[2 * N]; // rotations for fft
	float* pA;
	float* pRot;

	// fill arrays with data
	for (int i = 0; i < N; i++)
	{
		A[2 * i] = static_cast<float>(i);
		A[2 * i + 1] = static_cast<float>(i);
	}
	// calc "twiddle factors" on cpu
	for (int i = 0; i < (N); i++)
	{
		Rot[2 * i] = cosf((i * (6.2857f)) / N);
		Rot[2 * i + 1] = sinf((i * (6.2857f)) / N);
	}
	cudaMalloc((void**)&pA, nBytes); // (dst, size)
	cudaMalloc((void**)&pRot, nBytes);
	cudaMemcpy(pA, A, nBytes, cudaMemcpyHostToDevice); // (dst, src, size, type)
	cudaMemcpy(pRot, Rot, nBytes, cudaMemcpyHostToDevice);

	//__global__ functions are called: Func<<< Dg, Db, Ns >>>(parameter);
	fft<<<1, D>>>(pA, pRot); // single block with D threads

	cudaMemcpy(A, pA, nBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(Rot, pRot, nBytes, cudaMemcpyDeviceToHost);

	printf("The outputs are: \n");
	for (int l = 0; l < N; l++)
	{
		//printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);
	}

	printDeviceInfo();
}

void printDeviceInfo()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (MHz): %d\n",
			prop.memoryClockRate / 1024);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
		printf("  Total constant memory (bytes) %d\n", prop.totalConstMem);
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
		printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
	}
}