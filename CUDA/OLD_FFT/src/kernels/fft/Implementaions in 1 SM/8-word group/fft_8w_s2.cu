#define N 512
#define M 2*N
#define D N/4    
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SROT[N];
	short j = threadIdx.x;
	short n = logf(N) / logf(8);

	SROT[j] = ROT[j];
	SROT[j + blockDim.x] = ROT[j + blockDim.x];
	SROT[j + 2 * blockDim.x] = ROT[j + 2 * blockDim.x];
	SROT[j + 3 * blockDim.x] = ROT[j + 3 * blockDim.x];

	SA[j] = A[j];
	SA[j + blockDim.x] = A[j + blockDim.x];
	SA[j + 2 * blockDim.x] = A[j + 2 * blockDim.x];
	SA[j + 3 * blockDim.x] = A[j + 3 * blockDim.x];
	SA[j + 4 * blockDim.x] = A[j + 4 * blockDim.x];
	SA[j + 5 * blockDim.x] = A[j + 5 * blockDim.x];
	SA[j + 6 * blockDim.x] = A[j + 6 * blockDim.x];
	SA[j + 7 * blockDim.x] = A[j + 7 * blockDim.x];

	__syncthreads();

	short ind0, ind1, ind2, ind3;
	short r0, r1, r2, r3, r4, r5, ind_tmp, r_tmp;
	short i = j / 2;
	short k = j % 2;

	for (short s = 1; s <= n; s++)
	{
		short p = M / (1 << (3 * s));
		ind_tmp = 2 * (i + (i / (1 << 3 * (n - s))) * (1 << 3 * (n - s)) * 7);
		ind0 = ind_tmp + p * k;
		ind1 = ind_tmp + 2 * p + p * k;
		ind2 = ind_tmp + 4 * p + p * k;
		ind3 = ind_tmp + 6 * p + p * k;

		r_tmp = (i % (1 << 3 * (n - s))) * (1 << 3 * (s - 1));
		r0 = r_tmp + k * (N / 8);
		r1 = r_tmp + 2 * (N / 8) + k * (N / 8);
		r2 = 2 * r_tmp + k * (N / 4);
		r3 = r2;
		r4 = 4 * r_tmp;
		r5 = r4;
		/*SA[j] = ind0;
		SA[j+16] = ind1;
		SA[j+32] = ind2;
		SA[j+48] = ind3;*/


		SB[ind0] = SA[ind0] + SA[ind2];
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind2 + 1];
		SB[ind2] = SA[ind0] - SA[ind2];
		SB[ind2 + 1] = SA[ind0 + 1] - SA[ind2 + 1];

		SB[ind1] = SA[ind1] + SA[ind3];
		SB[ind1 + 1] = SA[ind1 + 1] + SA[ind3 + 1];
		SB[ind3] = SA[ind1] - SA[ind3];
		SB[ind3 + 1] = SA[ind1 + 1] - SA[ind3 + 1];

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];
		SA[ind2] = SB[ind2] * SROT[2 * r0] + SB[ind2 + 1] * SROT[2 * r0 + 1];
		SA[ind2 + 1] = -SB[ind2] * SROT[2 * r0 + 1] + SB[ind2 + 1] * SROT[2 * r0];

		SA[ind1] = SB[ind1];
		SA[ind1 + 1] = SB[ind1 + 1];
		SA[ind3] = SB[ind3] * SROT[2 * r1] + SB[ind3 + 1] * SROT[2 * r1 + 1];
		SA[ind3 + 1] = -SB[ind3] * SROT[2 * r1 + 1] + SB[ind3 + 1] * SROT[2 * r1];

		//2nd stage:
		SB[ind0] = SA[ind0] + SA[ind1];
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind1 + 1];
		SB[ind1] = SA[ind0] - SA[ind1];
		SB[ind1 + 1] = SA[ind0 + 1] - SA[ind1 + 1];

		SB[ind2] = SA[ind2] + SA[ind3];
		SB[ind2 + 1] = SA[ind2 + 1] + SA[ind3 + 1];
		SB[ind3] = SA[ind2] - SA[ind3];
		SB[ind3 + 1] = SA[ind2 + 1] - SA[ind3 + 1];

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];
		SA[ind1] = SB[ind1] * SROT[2 * r2] + SB[ind1 + 1] * SROT[2 * r2 + 1];
		SA[ind1 + 1] = -SB[ind1] * SROT[2 * r2 + 1] + SB[ind1 + 1] * SROT[2 * r2];

		SA[ind2] = SB[ind2];
		SA[ind2 + 1] = SB[ind2 + 1];
		SA[ind3] = SB[ind3] * SROT[2 * r3] + SB[ind3 + 1] * SROT[2 * r3 + 1];
		SA[ind3 + 1] = -SB[ind3] * SROT[2 * r3 + 1] + SB[ind3 + 1] * SROT[2 * r3];

		//__syncthreads();

		//3rd stage:
		short tmp_s3 = ind_tmp + 2 * p * k;

		SB[tmp_s3] = SA[tmp_s3] + SA[tmp_s3 + p];
		SB[tmp_s3 + 1] = SA[tmp_s3 + 1] + SA[tmp_s3 + p + 1];
		SB[tmp_s3 + p] = SA[tmp_s3] - SA[tmp_s3 + p];
		SB[tmp_s3 + p + 1] = SA[tmp_s3 + 1] - SA[tmp_s3 + p + 1];

		SB[tmp_s3 + 4 * p] = SA[tmp_s3 + 4 * p] + SA[tmp_s3 + 5 * p];
		SB[tmp_s3 + 4 * p + 1] = SA[tmp_s3 + 4 * p + 1] + SA[tmp_s3 + 5 * p + 1];
		SB[tmp_s3 + 5 * p] = SA[tmp_s3 + 4 * p] - SA[tmp_s3 + 5 * p];
		SB[tmp_s3 + 5 * p + 1] = SA[tmp_s3 + 4 * p + 1] - SA[tmp_s3 + 5 * p + 1];

		SA[tmp_s3] = SB[tmp_s3];
		SA[tmp_s3 + 1] = SB[tmp_s3 + 1];
		SA[tmp_s3 + p] = SB[tmp_s3 + p] * SROT[2 * r4] + SB[tmp_s3 + p + 1] * SROT[2 * r4 + 1];
		SA[tmp_s3 + p + 1] = -SB[tmp_s3 + p] * SROT[2 * r4 + 1] + SB[tmp_s3 + p + 1] * SROT[2 * r4];

		SA[tmp_s3 + 4 * p] = SB[tmp_s3 + 4 * p];
		SA[tmp_s3 + 4 * p + 1] = SB[tmp_s3 + 4 * p + 1];
		SA[tmp_s3 + 5 * p] = SB[tmp_s3 + 5 * p] * SROT[2 * r5] + SB[tmp_s3 + 5 * p + 1] * SROT[2 * r5 + 1];
		SA[tmp_s3 + 5 * p + 1] = -SB[tmp_s3 + 5 * p] * SROT[2 * r5 + 1] + SB[tmp_s3 + 5 * p + 1] * SROT[2 * r5];

	}

	A[j] = SA[j];
	A[j + blockDim.x] = SA[j + blockDim.x];
	A[j + 2 * blockDim.x] = SA[j + 2 * blockDim.x];
	A[j + 3 * blockDim.x] = SA[j + 3 * blockDim.x];
	A[j + 4 * blockDim.x] = SA[j + 4 * blockDim.x];
	A[j + 5 * blockDim.x] = SA[j + 5 * blockDim.x];
	A[j + 6 * blockDim.x] = SA[j + 6 * blockDim.x];
	A[j + 7 * blockDim.x] = SA[j + 7 * blockDim.x];

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

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}






