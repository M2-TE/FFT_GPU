#define N 512 // N complex numbers
#define M 2*N // M 32-bit floats (real, imaginary)
#define D N/8 // D number of threads (one block), each thread processing an 8-point fft   
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SROT[N];
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

	/*
	0R	0
	0I	1
	1R	2
	1I	3
		...
	7I	15

	//1st stage:
	temp = SA[0] + SA[8];
	SA[8] = SA[0] - SA[8];
	SA[0] = temp;

	...
	and so on

	temp = SA[4] + SA[12];
	// simplified here on phi=2
	*/

	__syncthreads();

	short ind0, ind1, ind2, ind3, ind4, ind5, ind6, ind7;
	short r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

	for (short s = 1; s <= n; s++)
	{
		short p = M / (1 << (3 * s));
		ind0 = 2 * (tid + (tid / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);
		ind1 = ind0 + p;
		ind2 = ind1 + p;//ind0+2p
		ind3 = ind2 + p;//ind0+3p
		ind4 = ind3 + p;//ind0+4p
		ind5 = ind4 + p;//ind0+5p
		ind6 = ind5 + p;//ind0+6p
		ind7 = ind6 + p;//ind0+7p

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
		/*SA[tid]    = ind0;
		SA[tid+8]  = ind1;
		SA[tid+16] = ind2;
		SA[tid+24] = ind3;
		SA[tid+32] = ind4;
		SA[tid+40] = ind5;
		SA[tid+48] = ind6;
		SA[tid+54] = ind7;*/

		/*SA[tid]    = r0;
		SA[tid+8]  = r1;
		SA[tid+16] = r2;
		SA[tid+24] = r3;
		SA[tid+32] = r4;
		SA[tid+40] = r5;
		SA[tid+48] = r6;
		SA[tid+56] = r7;
		SA[tid+64] = r8;
		SA[tid+72] = r9;
		SA[tid+80] = r10;
		SA[tid+88] = r11;*/

		//1st stage:
		SB[ind0] = SA[ind0] + SA[ind4];
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind4 + 1];
		SB[ind4] = SA[ind0] - SA[ind4];
		SB[ind4 + 1] = SA[ind0 + 1] - SA[ind4 + 1];

		SB[ind1] = SA[ind1] + SA[ind5];
		SB[ind1 + 1] = SA[ind1 + 1] + SA[ind5 + 1];
		SB[ind5] = SA[ind1] - SA[ind5];
		SB[ind5 + 1] = SA[ind1 + 1] - SA[ind5 + 1];

		SB[ind2] = SA[ind2] + SA[ind6];
		SB[ind2 + 1] = SA[ind2 + 1] + SA[ind6 + 1];
		SB[ind6] = SA[ind2] - SA[ind6];
		SB[ind6 + 1] = SA[ind2 + 1] - SA[ind6 + 1];

		SB[ind3] = SA[ind3] + SA[ind7];
		SB[ind3 + 1] = SA[ind3 + 1] + SA[ind7 + 1];
		SB[ind7] = SA[ind3] - SA[ind7];
		SB[ind7 + 1] = SA[ind3 + 1] - SA[ind7 + 1];

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];
		SA[ind4] = SB[ind4] * SROT[2 * r0] + SB[ind4 + 1] * SROT[2 * r0 + 1];
		SA[ind4 + 1] = -SB[ind4] * SROT[2 * r0 + 1] + SB[ind4 + 1] * SROT[2 * r0];

		SA[ind1] = SB[ind1];
		SA[ind1 + 1] = SB[ind1 + 1];
		SA[ind5] = SB[ind5] * SROT[2 * r1] + SB[ind5 + 1] * SROT[2 * r1 + 1];
		SA[ind5 + 1] = -SB[ind5] * SROT[2 * r1 + 1] + SB[ind5 + 1] * SROT[2 * r1];

		SA[ind2] = SB[ind2];
		SA[ind2 + 1] = SB[ind2 + 1];
		SA[ind6] = SB[ind6] * SROT[2 * r2] + SB[ind6 + 1] * SROT[2 * r2 + 1];
		SA[ind6 + 1] = -SB[ind6] * SROT[2 * r2 + 1] + SB[ind6 + 1] * SROT[2 * r2];

		SA[ind3] = SB[ind3];
		SA[ind3 + 1] = SB[ind3 + 1];
		SA[ind7] = SB[ind7] * SROT[2 * r3] + SB[ind7 + 1] * SROT[2 * r3 + 1];
		SA[ind7 + 1] = -SB[ind7] * SROT[2 * r3 + 1] + SB[ind7 + 1] * SROT[2 * r3];
		__syncthreads();

		//2nd stage:
		SB[ind0] = SA[ind0] + SA[ind2];
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind2 + 1];
		SB[ind2] = SA[ind0] - SA[ind2];
		SB[ind2 + 1] = SA[ind0 + 1] - SA[ind2 + 1];

		SB[ind1] = SA[ind1] + SA[ind3];
		SB[ind1 + 1] = SA[ind1 + 1] + SA[ind3 + 1];
		SB[ind3] = SA[ind1] - SA[ind3];
		SB[ind3 + 1] = SA[ind1 + 1] - SA[ind3 + 1];

		SB[ind4] = SA[ind4] + SA[ind6];
		SB[ind4 + 1] = SA[ind4 + 1] + SA[ind6 + 1];
		SB[ind6] = SA[ind4] - SA[ind6];
		SB[ind6 + 1] = SA[ind4 + 1] - SA[ind6 + 1];

		SB[ind5] = SA[ind5] + SA[ind7];
		SB[ind5 + 1] = SA[ind5 + 1] + SA[ind7 + 1];
		SB[ind7] = SA[ind5] - SA[ind7];
		SB[ind7 + 1] = SA[ind5 + 1] - SA[ind7 + 1];

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];
		SA[ind2] = SB[ind2] * SROT[2 * r4] + SB[ind2 + 1] * SROT[2 * r4 + 1];
		SA[ind2 + 1] = -SB[ind2] * SROT[2 * r4 + 1] + SB[ind2 + 1] * SROT[2 * r4];

		SA[ind1] = SB[ind1];
		SA[ind1 + 1] = SB[ind1 + 1];
		SA[ind3] = SB[ind3] * SROT[2 * r5] + SB[ind3 + 1] * SROT[2 * r5 + 1];
		SA[ind3 + 1] = -SB[ind3] * SROT[2 * r5 + 1] + SB[ind3 + 1] * SROT[2 * r5];

		SA[ind4] = SB[ind4];
		SA[ind4 + 1] = SB[ind4 + 1];
		SA[ind6] = SB[ind6] * SROT[2 * r6] + SB[ind6 + 1] * SROT[2 * r6 + 1];
		SA[ind6 + 1] = -SB[ind6] * SROT[2 * r6 + 1] + SB[ind6 + 1] * SROT[2 * r6];

		SA[ind5] = SB[ind5];
		SA[ind5 + 1] = SB[ind5 + 1];
		SA[ind7] = SB[ind7] * SROT[2 * r7] + SB[ind7 + 1] * SROT[2 * r7 + 1];
		SA[ind7 + 1] = -SB[ind7] * SROT[2 * r7 + 1] + SB[ind7 + 1] * SROT[2 * r7];
		__syncthreads();
		//3rd stage:
		SB[ind0] = SA[ind0] + SA[ind1];
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind1 + 1];
		SB[ind1] = SA[ind0] - SA[ind1];
		SB[ind1 + 1] = SA[ind0 + 1] - SA[ind1 + 1];

		SB[ind2] = SA[ind2] + SA[ind3];
		SB[ind2 + 1] = SA[ind2 + 1] + SA[ind3 + 1];
		SB[ind3] = SA[ind2] - SA[ind3];
		SB[ind3 + 1] = SA[ind2 + 1] - SA[ind3 + 1];

		SB[ind4] = SA[ind4] + SA[ind5];
		SB[ind4 + 1] = SA[ind4 + 1] + SA[ind5 + 1];
		SB[ind5] = SA[ind4] - SA[ind5];
		SB[ind5 + 1] = SA[ind4 + 1] - SA[ind5 + 1];

		SB[ind6] = SA[ind6] + SA[ind7];
		SB[ind6 + 1] = SA[ind6 + 1] + SA[ind7 + 1];
		SB[ind7] = SA[ind6] - SA[ind7];
		SB[ind7 + 1] = SA[ind6 + 1] - SA[ind7 + 1];

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];
		SA[ind1] = SB[ind1] * SROT[2 * r8] + SB[ind1 + 1] * SROT[2 * r8 + 1];
		SA[ind1 + 1] = -SB[ind1] * SROT[2 * r8 + 1] + SB[ind1 + 1] * SROT[2 * r8];

		SA[ind2] = SB[ind2];
		SA[ind2 + 1] = SB[ind2 + 1];
		SA[ind3] = SB[ind3] * SROT[2 * r9] + SB[ind3 + 1] * SROT[2 * r9 + 1];
		SA[ind3 + 1] = -SB[ind3] * SROT[2 * r9 + 1] + SB[ind3 + 1] * SROT[2 * r9];

		SA[ind4] = SB[ind4];
		SA[ind4 + 1] = SB[ind4 + 1];
		SA[ind5] = SB[ind5] * SROT[2 * r10] + SB[ind5 + 1] * SROT[2 * r10 + 1];
		SA[ind5 + 1] = -SB[ind5] * SROT[2 * r10 + 1] + SB[ind5 + 1] * SROT[2 * r10];

		SA[ind6] = SB[ind6];
		SA[ind6 + 1] = SB[ind6 + 1];
		SA[ind7] = SB[ind7] * SROT[2 * r11] + SB[ind7 + 1] * SROT[2 * r11 + 1];
		SA[ind7 + 1] = -SB[ind7] * SROT[2 * r11 + 1] + SB[ind7 + 1] * SROT[2 * r11];
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


#include  <stdio.h> 
#include  <math.h>
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

	//printf("The  outputs are: \n");
	//for (int l = 0; l < N; l++)
	//	printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}
