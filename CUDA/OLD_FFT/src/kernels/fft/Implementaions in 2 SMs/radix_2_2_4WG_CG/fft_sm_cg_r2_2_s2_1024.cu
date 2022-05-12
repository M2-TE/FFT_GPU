	#define N 1024
	#define M 2*N
	#define D N 

	__global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[N],SB[N],SROT[M];
	short j   = threadIdx.x;
	short n = logf(N)/logf(2);
	short i =j>>1;
	SA[i] = A[i+blockIdx.x*(N>>1)];
	SA[i+(N>>1)] = A[i+blockIdx.x*(N>>1)+N];

	SROT[j] = ROT[j];
	SROT[j+blockDim.x] = ROT[j+blockDim.x];

	__syncthreads();

	short k = j%2;
	short l = i>>1;
	short m = i%2;
	short r2 = m*(i>>(n-2));
	short signk = -(k<<1) + 1;
	short signm =  -(m<<1) +1;
	short ind0 = l<<1;
	short ind1 = (l<<1)+(N>>1);
	short s;
	
	short g = i+blockIdx.x*(blockDim.x>>1);
	
	//stage1:
	short r0 = (g%2)*(g>>(n-1));
	short signkr0 = -2*((!k)*r0)+1;
	short x = r0*(signk);
	SB[ind0+m*(D>>1)+k+x] = signkr0*(SA[ind0+k] + signm*SA[ind1+k]);
	
	__syncthreads();
	A[i+blockIdx.x*(N>>1)] = SB[i+(!blockIdx.x)*(N>>1)];
	__syncthreads();
	SB[i+(!blockIdx.x)*(N>>1)] = A[i+(!blockIdx.x)*(N>>1)];
	__syncthreads();

	//stage2:
	s=2;	
	short r1 = (((g%2)<<1) + ((g>>(n-s+1))%2))*((g%(1<<(n-s+1)))>>1);

	SA[j] = SB[ind0+k] + signm*SB[ind1+k];
	__syncthreads();
	SB[j] = signk*SA[i<<1]*SROT[(r1<<1)+k] + SA[(i<<1)+1]*SROT[(r1<<1)+(!k)];
	__syncthreads();

	//stage3:
	short signkr2 = -2*((!k)*r2)+1;
	short y = r2*(signk);
	SA[j+y] = signkr2*(SB[ind0+k] + signm*SB[ind1+k]);
	__syncthreads();

	//stage4:
	s=4;
	short r3 = (1<<(s-2))*(((i%2)<<1) + ((i>>1)%2))*(i>>(s-1));
	SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	//__syncthreads();
	SA[j] = signk*SB[i<<1]*SROT[(r3<<1)+k] + SB[(i<<1)+1]*SROT[(r3<<1)+(!k)];
	__syncthreads();
	
	//stage5:
	SB[j+y] = signkr2*(SA[ind0+k] + signm*SA[ind1+k]);
	__syncthreads();

	//stage6:
	s=6;
	short r5 = (1<<(s-2))*(((i%2)<<1) + ((i>>1)%2))*(i>>(s-1));
	SA[j] = SB[ind0+k] + signm*SB[ind1+k];
	//__syncthreads();
	SB[j] = signk*SA[i<<1]*SROT[(r5<<1)+k] + SA[(i<<1)+1]*SROT[(r5<<1)+(!k)];
	__syncthreads();

	//stage7:
	SA[j+y] = signkr2*(SB[ind0+k] + signm*SB[ind1+k]);
	__syncthreads();
	//stage8:
	s=8;
	short r7 = (1<<(s-2))*(((i%2)<<1) + ((i>>1)%2))*(i>>(s-1));
	SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	//__syncthreads();
	SA[j] = signk*SB[i<<1]*SROT[(r7<<1)+k] + SB[(i<<1)+1]*SROT[(r7<<1)+(!k)];
	__syncthreads();

	//stage9:
	SB[j+y] = signkr2*(SA[ind0+k] + signm*SA[ind1+k]);
	__syncthreads();
	//stage10:

	SA[j] = SB[ind0+k] + signm*SB[ind1+k];
	//SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	__syncthreads();

	A[i+blockIdx.x*(N)] = SA[i];
	A[i+(N>>1)+blockIdx.x*(N)] = SA[i+(N>>1)];
	

      }

       #include  <stdio.h> 
       #include  <math.h>
       int  main() 
       { 
        
           float A[2*N]; 
           float *Ad;
	   float ROT[2*N];
	   float *ROTd; 
	  
           int memsize= 2*N * sizeof(float); 
	   int rotsize = 2*N* sizeof(float);


		for(int i=0; i<N; i++)
		{
			A[2*i]  = i;
			A[2*i+1]= i;	
		}
		for(int j=0; j < (N); j++)
		{
			
			ROT[2*j]= cosf((j*(6.2857))/N);
			ROT[2*j+1]=sinf((j*(6.2857))/N);	
		}

           cudaMalloc((void**)&Ad, memsize);
	   cudaMalloc((void**)&ROTd, rotsize); 

           cudaMemcpy(Ad, A, memsize,  cudaMemcpyHostToDevice); 
	   cudaMemcpy(ROTd, ROT, rotsize,  cudaMemcpyHostToDevice); 

           //__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(2,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

     }








