#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 512 // N complex numbers
#define M 2*N // M 32-bit floats (real, imaginary)
#define D N/8 // D number of threads (one block), each thread processing an 8-point fft   
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SROT[N];
	short tid = threadIdx.x;
	short n = logf(N) / logf(8);

	SROT[tid] = ROT[tid];
	SROT[tid + blockDim.x] = ROT[tid + blockDim.x];
	SROT[tid + 2 * blockDim.x] = ROT[tid + 2 * blockDim.x];
	SROT[tid + 3 * blockDim.x] = ROT[tid + 3 * blockDim.x];
	SROT[tid + 4 * blockDim.x] = ROT[tid + 4 * blockDim.x];
	SROT[tid + 5 * blockDim.x] = ROT[tid + 5 * blockDim.x];
	SROT[tid + 6 * blockDim.x] = ROT[tid + 6 * blockDim.x];
	SROT[tid + 7 * blockDim.x] = ROT[tid + 7 * blockDim.x];

	SA[tid] = A[tid];
	SA[tid + blockDim.x] = A[tid + blockDim.x];
	SA[tid + 2 * blockDim.x] = A[tid + 2 * blockDim.x];
	SA[tid + 3 * blockDim.x] = A[tid + 3 * blockDim.x];
	SA[tid + 4 * blockDim.x] = A[tid + 4 * blockDim.x];
	SA[tid + 5 * blockDim.x] = A[tid + 5 * blockDim.x];
	SA[tid + 6 * blockDim.x] = A[tid + 6 * blockDim.x];
	SA[tid + 7 * blockDim.x] = A[tid + 7 * blockDim.x];
	SA[tid + 8 * blockDim.x] = A[tid + 8 * blockDim.x];
	SA[tid + 9 * blockDim.x] = A[tid + 9 * blockDim.x];
	SA[tid + 10 * blockDim.x] = A[tid + 10 * blockDim.x];
	SA[tid + 11 * blockDim.x] = A[tid + 11 * blockDim.x];
	SA[tid + 12 * blockDim.x] = A[tid + 12 * blockDim.x];
	SA[tid + 13 * blockDim.x] = A[tid + 13 * blockDim.x];
	SA[tid + 14 * blockDim.x] = A[tid + 14 * blockDim.x];
	SA[tid + 15 * blockDim.x] = A[tid + 15 * blockDim.x];

	__syncthreads();

	short r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

	for (short s = 1; s <= n; s++)
	{
		unsigned int p = M / (1 << (3 * s));
		unsigned int ind0 = 2 * (tid + (tid / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);

		// imaginary
		#define i0_R ind0
		#define i1_R ind0 +		p
		#define i2_R ind0 + 2 * p
		#define i3_R ind0 + 3 * p
		#define i4_R ind0 + 4 * p
		#define i5_R ind0 + 5 * p
		#define i6_R ind0 + 6 * p
		#define i7_R ind0 + 7 * p

		// real
		#define i0_I ind0		  + 1
		#define i1_I ind0 +		p + 1
		#define i2_I ind0 + 2 * p + 1
		#define i3_I ind0 + 3 * p + 1
		#define i4_I ind0 + 4 * p + 1
		#define i5_I ind0 + 5 * p + 1
		#define i6_I ind0 + 6 * p + 1
		#define i7_I ind0 + 7 * p + 1

		r0 = (tid % (1 << 3 * (n - s))) * (1 << 3 * (s - 1));
		r1 = r0 + (N / 8);
		r2 = r1 + (N / 8);//r0+2(N/8)
		r3 = r2 + (N / 8);//r0+3(N/8)
		r4 = 2 * r0;
		r5 = r4 + (N / 4);
		r6 = r4;
		r7 = r5;
		r8 = 2 * r4;
		r9 = r8;
		r10 = r8;
		r11 = r8;

		//1st stage:
		//float sbA_R = SA[i0_R] + SA[i4_R]; // real parts addition.
		//float sbA_I = SA[i0_I] + SA[i4_I]; //img parts addition.

		float sb_R; //real parts subtraction
		float sb_I; //img parts subtraction.

		sb_R = SA[i0_R] - SA[i4_R];
		sb_I = SA[i0_I] - SA[i4_I];
		SA[i0_R] = SA[i0_R] + SA[i4_R];
		SA[i0_I] = SA[i0_I] + SA[i4_I];
		SA[i4_R] = sb_R * SROT[2 * r0] + sb_I * SROT[2 * r0 + 1];
		SA[i4_I] = -sb_R * SROT[2 * r0 + 1] + sb_I * SROT[2 * r0];

		sb_R = SA[i1_R] - SA[i5_R];
		sb_I = SA[i1_I] - SA[i5_I];
		SA[i1_R] = SA[i1_R] + SA[i5_R];
		SA[i1_I] = SA[i1_I] + SA[i5_I];
		SA[i5_R] = sb_R * SROT[2 * r1] + sb_I * SROT[2 * r1 + 1];
		SA[i5_I] = -sb_R * SROT[2 * r1 + 1] + sb_I * SROT[2 * r1];

		sb_R = SA[i2_R] - SA[i6_R];
		sb_I = SA[i2_I] - SA[i6_I];
		SA[i2_R] = SA[i2_R] + SA[i6_R];
		SA[i2_I] = SA[i2_I] + SA[i6_I];
		SA[i6_R] = sb_R * SROT[2 * r2] + sb_I * SROT[2 * r2 + 1];
		SA[i6_I] = -sb_R * SROT[2 * r2 + 1] + sb_I * SROT[2 * r2];

		sb_R = SA[i3_R] - SA[i7_R];
		sb_I = SA[i3_I] - SA[i7_I];
		SA[i3_R] = SA[i3_R] + SA[i7_R];
		SA[i3_I] = SA[i3_I] + SA[i7_I];
		SA[i7_R] = sb_R * SROT[2 * r3] + sb_I * SROT[2 * r3 + 1];
		SA[i7_I] = -sb_R * SROT[2 * r3 + 1] + sb_I * SROT[2 * r3];
		__syncthreads();

		//2nd stage:
		sb_R = SA[i0_R] - SA[i2_R];
		sb_I = SA[i0_I] - SA[i2_I];
		SA[i0_R] = SA[i0_R] + SA[i2_R];
		SA[i0_I] = SA[i0_I] + SA[i2_I];
		SA[i2_R] = sb_R * SROT[2 * r4] + sb_I * SROT[2 * r4 + 1];
		SA[i2_I] = -sb_R * SROT[2 * r4 + 1] + sb_I * SROT[2 * r4];

		sb_R = SA[i1_R] - SA[i3_R];
		sb_I = SA[i1_I] - SA[i3_I];
		SA[i1_R] = SA[i1_R] + SA[i3_R];
		SA[i1_I] = SA[i1_I] + SA[i3_I];
		SA[i3_R] = sb_R * SROT[2 * r5] + sb_I * SROT[2 * r5 + 1];
		SA[i3_I] = -sb_R * SROT[2 * r5 + 1] + sb_I * SROT[2 * r5];

		sb_R = SA[i4_R] - SA[i6_R];
		sb_I = SA[i4_I] - SA[i6_I];
		SA[i4_R] = SA[i4_R] + SA[i6_R];
		SA[i4_I] = SA[i4_I] + SA[i6_I];
		SA[i6_R] = sb_R * SROT[2 * r6] + sb_I * SROT[2 * r6 + 1];
		SA[i6_I] = -sb_R * SROT[2 * r6 + 1] + sb_I * SROT[2 * r6];

		sb_R = SA[i5_R] - SA[i7_R];
		sb_I = SA[i5_I] - SA[i7_I];
		SA[i5_R] = SA[i5_R] + SA[i7_R];
		SA[i5_I] = SA[i5_I] + SA[i7_I];
		SA[i7_R] = sb_R * SROT[2 * r7] + sb_I * SROT[2 * r7 + 1];
		SA[i7_I] = -sb_R * SROT[2 * r7 + 1] + sb_I * SROT[2 * r7];
		__syncthreads();

		//3rd stage:
		sb_R = SA[i0_R] - SA[i1_R];
		sb_I = SA[i0_I] - SA[i1_I];
		SA[i0_R] = SA[i0_R] + SA[i1_R];
		SA[i0_I] = SA[i0_I] + SA[i1_I];
		SA[i1_R] = sb_R * SROT[2 * r8] + sb_I * SROT[2 * r8 + 1];
		SA[i1_I] = -sb_R * SROT[2 * r8 + 1] + sb_I * SROT[2 * r8];

		sb_R = SA[i2_R] - SA[i3_R];
		sb_I = SA[i2_I] - SA[i3_I];
		SA[i2_R] = SA[i2_R] + SA[i3_R];
		SA[i2_I] = SA[i2_I] + SA[i3_I];
		SA[i3_R] = sb_R * SROT[2 * r9] + sb_I * SROT[2 * r9 + 1];
		SA[i3_I] = -sb_R * SROT[2 * r9 + 1] + sb_I * SROT[2 * r9];

		sb_R = SA[i4_R] - SA[i5_R];
		sb_I = SA[i4_I] - SA[i5_I];
		SA[i4_R] = SA[i4_R] + SA[i5_R];
		SA[i4_I] = SA[i4_I] + SA[i5_I];
		SA[i5_R] = sb_R * SROT[2 * r10] + sb_I * SROT[2 * r10 + 1];
		SA[i5_I] = -sb_R * SROT[2 * r10 + 1] + sb_I * SROT[2 * r10];

		sb_R = SA[i6_R] - SA[i7_R];
		sb_I = SA[i6_I] - SA[i7_I];
		SA[i6_R] = SA[i6_R] + SA[i7_R];
		SA[i6_I] = SA[i6_I] + SA[i7_I];
		SA[i7_R] = sb_R * SROT[2 * r11] + sb_I * SROT[2 * r11 + 1];
		SA[i7_I] = -sb_R * SROT[2 * r11 + 1] + sb_I * SROT[2 * r11];
		__syncthreads();

	}

	A[tid] = SA[tid];
	A[tid + blockDim.x] = SA[tid + blockDim.x];
	A[tid + 2 * blockDim.x] = SA[tid + 2 * blockDim.x];
	A[tid + 3 * blockDim.x] = SA[tid + 3 * blockDim.x];
	A[tid + 4 * blockDim.x] = SA[tid + 4 * blockDim.x];
	A[tid + 5 * blockDim.x] = SA[tid + 5 * blockDim.x];
	A[tid + 6 * blockDim.x] = SA[tid + 6 * blockDim.x];
	A[tid + 7 * blockDim.x] = SA[tid + 7 * blockDim.x];
	A[tid + 8 * blockDim.x] = SA[tid + 8 * blockDim.x];
	A[tid + 9 * blockDim.x] = SA[tid + 9 * blockDim.x];
	A[tid + 10 * blockDim.x] = SA[tid + 10 * blockDim.x];
	A[tid + 11 * blockDim.x] = SA[tid + 11 * blockDim.x];
	A[tid + 12 * blockDim.x] = SA[tid + 12 * blockDim.x];
	A[tid + 13 * blockDim.x] = SA[tid + 13 * blockDim.x];
	A[tid + 14 * blockDim.x] = SA[tid + 14 * blockDim.x];
	A[tid + 15 * blockDim.x] = SA[tid + 15 * blockDim.x];
	__syncthreads();
}


int  main()
{

	float A[2 * N];
	float* Ad;
	float ROT[N];
	float* ROTd;

	int memsize = 2 * N * sizeof(float);
	int rotsize = N * sizeof(float);


	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}
	for (int j = 0; j < (N / 2); j++)
	{

		ROT[2 * j] = cosf((j * (6.2857)) / N);
		ROT[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	cudaMalloc((void**)&Ad, memsize);
	cudaMalloc((void**)&ROTd, rotsize);

	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(ROTd, ROT, rotsize, cudaMemcpyHostToDevice);

	// __global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
	dim3 gridDim(1, 1);
	dim3 blockDim(D, 1);
	fft << <gridDim, blockDim >> > (Ad, ROTd);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);


	// testing out the weirdass formulas
	int tid = 63;
	int n = logf(N) / logf(8);
	for (int s = 1; s <= n; s++) {

		int b = 2 * (tid + (tid / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);
		int p = M / (1 << (3 * s));

		printf("n: %d, index: %d, step: %d\n", b + 0 * p, b, p);
	}

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}
