#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 512 // N complex numbers
//#define N 4096 // N complex numbers
#define M 2*N // M 32-bit floats (real, imaginary)
#define D N/8 // D number of threads (one block), each thread processing an 8-point fft   
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SROT[N];
	unsigned int tid = threadIdx.x;
	unsigned int n = logf(N) / logf(8);

	// transfer rotations from global to shared memory
	SROT[tid] = ROT[tid];
	SROT[tid + blockDim.x] = ROT[tid + blockDim.x];
	for (unsigned int i = 2; i < 8; i++) SROT[tid + i * blockDim.x] = ROT[tid + i * blockDim.x];

	// transfer samples from global to shared memory
	SA[tid] = A[tid];
	SA[tid + blockDim.x] = A[tid + blockDim.x];
	for (unsigned int i = 2; i < 16; i++) SA[tid + i * blockDim.x] = A[tid + i * blockDim.x];

	__syncthreads();

	unsigned int r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

	for (short s = 1; s <= n; s++)
	{
		unsigned int p = M / (1 << (3 * s));
		unsigned int ind0 = 2 * (tid + (tid / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);

		// imaginary
		#define i0_R ind0
		#define i1_R ind0 +	p
		#define i2_R ind0 + p * 2 // do the numbers 2 to 7 consume a register slot?
		#define i3_R ind0 + p * 3
		#define i4_R ind0 + p * 4
		#define i5_R ind0 + p * 5
		#define i6_R ind0 + p * 6
		#define i7_R ind0 + p * 7

		// real
		#define i0_I ind0 + 1
		#define i1_I ind0 + 1 +	p
		#define i2_I ind0 + 1 + p * 2
		#define i3_I ind0 + 1 + p * 3
		#define i4_I ind0 + 1 + p * 4
		#define i5_I ind0 + 1 + p * 5
		#define i6_I ind0 + 1 + p * 6
		#define i7_I ind0 + 1 + p * 7

		r0 = (tid % (1 << 3 * (n - s))) * (1 << 3 * (s - 1));
		r1 = r0 + (N / 8);
		r2 = r1 + (N / 8);//r0+2(N/8)
		r3 = r2 + (N / 8);//r0+3(N/8)

		r4 = 2 * r0; // trivial rotation?
		r5 = r4 + (N / 4); // werent these special too somehow?
		r6 = r4; // trivial rotation?
		r7 = r5;

		// all trivial rotations?
		r8 = 2 * r4;
		r9 = r8;
		r10 = r8;
		r11 = r8;

		float sb_R; //real parts subtraction
		float sb_I; //img parts subtraction.

		// TODO: forgo use of shared memory entirely outside of initial read!
		// TODO: can simplify some rotations

		//1st stage:
		{
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
		}
		//__syncthreads(); // is this even needed?

		//2nd stage:
		{
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
		}
		//__syncthreads(); // is this even needed?

		//3rd stage:
		{
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
		}
		__syncthreads();
	}

	A[tid] = SA[tid];
	A[tid + blockDim.x] = SA[tid + blockDim.x];
	for (unsigned int i = 2; i < 16; i++) A[tid + i * blockDim.x] = SA[tid + i * blockDim.x];
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
	int tid = 14;
	int n = logf(N) / logf(8);
	for (int s = 1; s <= n; s++) {

		int b = 2 * (tid + (tid / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);
		int p = M / (1 << (3 * s));

		int r0 = (tid % (1 << 3 * (n - s))) * (1 << 3 * (s - 1));
		int r1 = r0 + (N / 8);
		int r2 = r1 + (N / 8);//r0+2(N/8)
		int r3 = r2 + (N / 8);//r0+3(N/8)
		int r4 = 2 * r0;
		int r5 = r4 + (N / 4);
		int r6 = r4;
		int r7 = r5;
		int r8 = 2 * r4;
		int r9 = r8;
		int r10 = r8;
		int r11 = r8;

		printf("n: %d, index: %d, step: %d\n", b + 0 * p, b, p);
		printf("n: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11);
		printf("n: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", ROT[r0 * 2], ROT[r1 * 2], ROT[r2 * 2], ROT[r3 * 2], ROT[r4 * 2], ROT[r5 * 2], ROT[r6 * 2], ROT[r7 * 2], ROT[r8 * 2], ROT[r9 * 2], ROT[r10 * 2], ROT[r11 * 2]);
		printf("n: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", ROT[r0 * 2 + 1], ROT[r1 * 2 + 1], ROT[r2 * 2 + 1], ROT[r3 * 2 + 1], ROT[r4 * 2 + 1], ROT[r5 * 2 + 1], ROT[r6 * 2 + 1], ROT[r7 * 2 + 1], ROT[r8 * 2 + 1], ROT[r9 * 2 + 1], ROT[r10 * 2 + 1], ROT[r11 * 2 + 1]);
		//printf("n: %f, %f, %f, %f\n", cosf((r0 * (6.2857)) / N), cosf((r1 * 0.5f * (6.2857)) / N), ROT[r2], ROT[r3]);
		printf("\n");
	}

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++) {
		//printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);
	}

}
