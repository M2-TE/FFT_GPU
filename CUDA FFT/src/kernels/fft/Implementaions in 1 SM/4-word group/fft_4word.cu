        #define N 16
	#define M 2*N
	#define D N/4    
       __global__  void fft(float* A , float* ROT ) 
       { 
	__shared__ float SA[M],SB[M],SROT[N];
	short i   = threadIdx.x;
	short n = logf(N)/logf(4);

	SROT[i] = ROT[i];
	SROT[i+blockDim.x] = ROT[i+blockDim.x];
	SROT[i+2*blockDim.x] = ROT[i+2*blockDim.x];
	SROT[i+3*blockDim.x] = ROT[i+3*blockDim.x];

	SA[i]= A[i];
	SA[i+blockDim.x]= A[i+blockDim.x];
	SA[i+2*blockDim.x] = A[i+2*blockDim.x];
	SA[i+3*blockDim.x] = A[i+3*blockDim.x];
	SA[i+4*blockDim.x] = A[i+4*blockDim.x];
	SA[i+5*blockDim.x] = A[i+5*blockDim.x];
	SA[i+6*blockDim.x] = A[i+6*blockDim.x];
	SA[i+7*blockDim.x] = A[i+7*blockDim.x];	
	
	__syncthreads();
	
	short ind0, ind1, ind2, ind3;
	short r0,r1,r2,r3;
	
	for(short s=1; s<=n; s++)
	{
		short p = M/(1<<(2*s));
		ind0 = 2*(i+(i/(1<<2*(n-s)))*(1<<2*(n-s))*3);
		ind1 = ind0+p;
		ind2 = ind1+p;//ind0+2p
		ind3 = ind2+p;//ind0+3p

		r0 = (i%(1<<2*(n-s)))*(1<<2*(s-1));
		r1 = r0+(N/4);
		r2 = 2*r0;
		r3 =r2;

		SB[ind0]   =  SA[ind0] + SA[ind2];
		SB[ind0+1] =  SA[ind0+1] + SA[ind2+1]; 
		SB[ind2]   =  SA[ind0] - SA[ind2];
		SB[ind2+1] =  SA[ind0+1] - SA[ind2+1]; 

		SB[ind1]   =  SA[ind1] + SA[ind3];
		SB[ind1+1] =  SA[ind1+1] + SA[ind3+1]; 
		SB[ind3]   =  SA[ind1] - SA[ind3];
		SB[ind3+1] =  SA[ind1+1] - SA[ind3+1];	

		SA[ind0]   =  SB[ind0];
		SA[ind0+1] =  SB[ind0+1];
		SA[ind2]   =  SB[ind2]*SROT[2*r0] + SB[ind2+1]*SROT[2*r0+1];
		SA[ind2+1] =  -SB[ind2]*SROT[2*r0+1] + SB[ind2+1]*SROT[2*r0];

		SA[ind1]   =  SB[ind1];
		SA[ind1+1] =  SB[ind1+1];
		SA[ind3]   =  SB[ind3]*SROT[2*r1] + SB[ind3+1]*SROT[2*r1+1];
		SA[ind3+1] =  -SB[ind3]*SROT[2*r1+1] + SB[ind3+1]*SROT[2*r1];

		SB[ind0]   =  SA[ind0] + SA[ind1];
		SB[ind0+1] =  SA[ind0+1] + SA[ind1+1]; 
		SB[ind1]   =  SA[ind0] - SA[ind1];
		SB[ind1+1] =  SA[ind0+1] - SA[ind1+1]; 
		
		SB[ind2]   =  SA[ind2] + SA[ind3];
		SB[ind2+1] =  SA[ind2+1] + SA[ind3+1]; 
		SB[ind3] =  SA[ind2] - SA[ind3];
		SB[ind3+1]   =  SA[ind2+1] - SA[ind3+1];

		SA[ind0]   =  SB[ind0];
		SA[ind0+1] =  SB[ind0+1];
		SA[ind1]   =  SB[ind1]*SROT[2*r2] + SB[ind1+1]*SROT[2*r2+1];
		SA[ind1+1] =  -SB[ind1]*SROT[2*r2+1] + SB[ind1+1]*SROT[2*r2];

		SA[ind2]   =  SB[ind2];
		SA[ind2+1] =  SB[ind2+1];
		SA[ind3]   =  SB[ind3]*SROT[2*r3] + SB[ind3+1]*SROT[2*r3+1];
		SA[ind3+1] =  -SB[ind3]*SROT[2*r3+1] + SB[ind3+1]*SROT[2*r3];

	}
	
	A[i] = SA[i];
	A[i+blockDim.x] = SA[i+blockDim.x];
	A[i+2*blockDim.x] = SA[i+2*blockDim.x];
	A[i+3*blockDim.x] = SA[i+3*blockDim.x];
	A[i+4*blockDim.x] = SA[i+4*blockDim.x];
	A[i+5*blockDim.x] = SA[i+5*blockDim.x];
	A[i+6*blockDim.x] = SA[i+6*blockDim.x];
	A[i+7*blockDim.x] = SA[i+7*blockDim.x];
	__syncthreads();
	}


       #include  <stdio.h> 
       #include  <math.h>
       int  main() 
       { 
        
           float A[2*N]; 
           float *Ad;
	   float ROT[N];
	   float *ROTd; 
	  
           int memsize= 2*N * sizeof(float); 
	   int rotsize = N* sizeof(float);


		for(int i=0; i<N; i++)
		{
			A[2*i]  = i;
			A[2*i+1]= i;	
		}
		for(int j=0; j < (N/2); j++)
		{
			
			ROT[2*j]= cosf((j*(6.2857))/N);
			ROT[2*j+1]=sinf((j*(6.2857))/N);	
		}

           cudaMalloc((void**)&Ad, memsize);
	   cudaMalloc((void**)&ROTd, rotsize); 

           cudaMemcpy(Ad, A, memsize,  cudaMemcpyHostToDevice); 
	   cudaMemcpy(ROTd, ROT, rotsize,  cudaMemcpyHostToDevice); 

           // __global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(1,1);
           dim3 blockDim(D,1);
	   fft<<<gridDim , blockDim>>>(Ad,ROTd );
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);

            printf("The  outputs are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 

      }
