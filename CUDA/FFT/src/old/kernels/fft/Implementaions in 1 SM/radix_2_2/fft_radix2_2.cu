#define N 64
#define M 2*N
#define D N   
__global__  void fft(float* A, float* ROT)
{
	__shared__ float SA[M], SB[M], SROT[M];
	short j = threadIdx.x;
	short n = logf(N) / logf(4);
	short n1 = logf(N) / logf(2);
	SROT[j] = ROT[j];
	SROT[j + blockDim.x] = ROT[j + blockDim.x];

	SA[j] = A[j];
	SA[j + blockDim.x] = A[j + blockDim.x];
	__syncthreads();

	short ind0, ind1, ind2, r1, r4, sign1, sign2, index1, index2, index3, index4, index5, index6;
	short i = j / 2;
	short k = j % 2;
	short h = i / 2;
	short l = j % 4;
	short m = i % 2;
	short w = m * k;

	//for iteration1:
	short s0 = 1;
	short p0 = M / (1 << (2 * s0));
	short ind_tmp0 = 2 * (h + (h / (1 << 2 * (n - s0))) * (1 << 2 * (n - s0)) * 3);

	ind0 = ind_tmp0 + l * p0;
	//stage1:
	sign1 = m * (-2) + 1;
	short signw = w * (-2) + 1;
	index1 = ind0 + sign1 * 2 * p0;

	SB[ind0 + w] = signw * (SA[index1] + sign1 * SA[ind0]);
	SB[ind0 + !w] = SA[index1 + 1] + sign1 * SA[ind0 + 1];

	//stage2:
	sign2 = k * (-2) + 1;
	//r1 =  (ind0/256)*((ind0>>1)%64) + k*((ind0/2)%64)*2;
	//r1 = (ind0/64)*((ind0/2)%16) + k*((ind0/2)%16)*2;
	//r1 = (ind0/16)*((ind0/2)%4) + k*((ind0/2)%4)*2;  

	s0 = 2;
	r1 = ((((ind0 >> (n1 - s0 + 1)) % 2) << 1) + ((ind0 >> (n1 - s0 + 2)))) * ((ind0 >> 1) % (1 << (n1 - s0)));
	index2 = ind0 + sign2 * p0;

	SA[ind0] = SB[index2] + sign2 * SB[ind0];
	SA[ind0 + 1] = SB[index2 + 1] + sign2 * SB[ind0 + 1];

	SB[ind0] = SA[ind0] * SROT[2 * r1] + SA[ind0 + 1] * SROT[2 * r1 + 1];
	SB[ind0 + 1] = -SA[ind0] * SROT[2 * r1 + 1] + SA[ind0 + 1] * SROT[2 * r1];
	__syncthreads();
	//for iteration2:
	short s1 = 2;
	short p1 = M / (1 << (2 * s1));
	short ind_tmp1 = 2 * (h + (h / (1 << 2 * (n - s1))) * (1 << 2 * (n - s1)) * 3);

	ind1 = ind_tmp1 + l * p1;

	//stage1:

	index3 = ind1 + sign1 * 2 * p1;

	SA[ind1 + w] = signw * (SB[index3] + sign1 * SB[ind1]);
	SA[ind1 + !w] = SB[index3 + 1] + sign1 * SB[ind1 + 1];

	//stage2:

	index4 = ind1 + sign2 * p1;
	s0 = 4;

	r4 = (1 << (s0 - 2)) * ((((ind1 >> (n1 - s0 + 1)) % 2) << 1) + ((ind1 >> (n1 - s0 + 2))) % 2) * ((ind1 >> 1) % (1 << (n1 - s0)));


	//r4 = (((ind1/16)%2)*4 + k*8*((ind1/8)%2))*((ind1/2)%4);

	SB[ind1] = SA[index4] + sign2 * SA[ind1];
	SB[ind1 + 1] = SA[index4 + 1] + sign2 * SA[ind1 + 1];

	SA[ind1] = SB[ind1] * SROT[2 * r4] + SB[ind1 + 1] * SROT[2 * r4 + 1];
	SA[ind1 + 1] = -SB[ind1] * SROT[2 * r4 + 1] + SB[ind1 + 1] * SROT[2 * r4];
	__syncthreads();
	//for iteration3:

	short s2 = 3;
	short p2 = M / (1 << (2 * s2));
	short ind_tmp2 = 2 * (h + (h / (1 << 2 * (n - s2))) * (1 << 2 * (n - s2)) * 3);

	ind2 = ind_tmp2 + l * p2;

	//for stage1:	
	index5 = ind2 + sign1 * 2 * p2;

	//for stage2:
	index6 = ind2 + sign2 * p2;

	SB[ind2 + w] = signw * (SA[index5] + sign1 * SA[ind2]);
	SB[ind2 + !w] = SA[index5 + 1] + sign1 * SA[ind2 + 1];
	SA[ind2] = SB[index6] + sign2 * SB[ind2];
	SA[ind2 + 1] = SB[index6 + 1] + sign2 * SB[ind2 + 1];

	//SA[j] = SB[j];
	//SA[j+blockDim.x] = SB[j+blockDim.x];
	__syncthreads();
	A[j] = SA[j];
	A[j + blockDim.x] = SA[j + blockDim.x];

}

#include  <stdio.h> 
#include  <math.h>
int  main()
{

	float A[2 * N];
	float* Ad;
	float ROT[2 * N];
	float* ROTd;

	int memsize = 2 * N * sizeof(float);
	int rotsize = 2 * N * sizeof(float);


	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}
	for (int j = 0; j < (N); j++)
	{

		ROT[2 * j] = cosf((j * (6.2857)) / N);
		ROT[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	cudaMalloc((void**)&Ad, memsize);
	cudaMalloc((void**)&ROTd, rotsize);

	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(ROTd, ROT, rotsize, cudaMemcpyHostToDevice);

	//__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
	dim3 gridDim(1, 1);
	dim3 blockDim(D, 1);
	fft << <gridDim, blockDim >> > (Ad, ROTd);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ROT, ROTd, rotsize, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}






