	#define N 64
	#define M 2*N
	#define D 2*N   
	#define n 6
       __global__  void fft(float* A , float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SROT[M];
	short j   = threadIdx.x;
	
	SROT[j] = ROT[j];

	SA[j] = A[j];
	__syncthreads();
	
    	short i = j>>1;
    	short k = j%2;
	short l = i>>1;
	short m = i%2;
	short r0 = m*(i>>(n-1));
	short signk = -(k<<1) + 1;
	short signm =  -(m<<1) +1;
	short x = r0*(signk);
	short signkr0 = -2*((!k)*r0)+1;
	short ind0 = l<<1;
	short ind1 = (l<<1)+N;
	short s;
        
	//stage1:
	SB[j+x] = signkr0*(SA[ind0+k] + signm*SA[ind1+k]);
	//__syncthreads();
	//stage2:
	s=2;
	short r1 = (((i%2)<<1) + ((i>>1)%2))*(i>>s); 
	SA[j] = SB[ind0+k] + signm*SB[ind1+k];
	SB[j] = signk*SA[i<<1]*SROT[(r1<<1)+k] + SA[(i<<1)+1]*SROT[(r1<<1)+(!k)];
	__syncthreads();
	//stage3:
	
	 SA[j+x] = signkr0*(SB[ind0+k] + signm*SB[ind1+k]);
	
	//stage4:
	 s=4;
	 short r3 = (1<<(s-2))*(((i%2)<<1) + ((i>>1)%2))*(i>>s);
	 SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	 SA[j] = signk*SB[i<<1]*SROT[(r3<<1)+k] + SB[(i<<1)+1]*SROT[(r3<<1)+(!k)];
	

	 //stage5:

	 //SB[j+x] = signkr0*(SA[ind0+k] + (signm)*SA[ind1+k]);

	  SB[j+x] = signkr0*(SA[ind0+k] + signm*(SA[ind1+k])) ;

	  
	 //stage6:
	 SA[j] = SB[ind0+k] + signm*SB[ind1+k];
	//SB[j] = SA[ind0+k] + signm*SA[ind1+k];
	
	__syncthreads();
	
	  A[j] = SA[j];

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
           dim3 gridDim(1,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

     }
		


	 






