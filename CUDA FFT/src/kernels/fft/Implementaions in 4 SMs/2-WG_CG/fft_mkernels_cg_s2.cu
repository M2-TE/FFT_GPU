	#define N  2048
	#define n 11
	#define M N/2
	#define D N/8
	__global__ void kernel0(float* A ,float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SROT[N];
	short j   = threadIdx.x;
	short b   = blockIdx.x;  
	//short n = logf(N)/logf(2);

	SA[j] = A[j+b*(D)];
	SA[j+D] = A[j+b*(D)+(N>>1)];
	SA[j+2*D] = A[j+b*(D)+N];
	SA[j+3*D] = A[j+b*(D)+N+(N>>1)];

	SROT[j] = ROT[j];
	SROT[j+D] = ROT[j+D];
	SROT[j+2*D] = ROT[j+2*D];
	SROT[j+3*D] = ROT[j+3*D];
	SROT[j+4*D] = ROT[j+4*D];
	SROT[j+5*D] = ROT[j+5*D];
	SROT[j+6*D] = ROT[j+6*D];
	SROT[j+7*D] = ROT[j+7*D];
	
	__syncthreads();
	//stage1:
	//short g = j+b*D;
	short r0 = j+(b<<(n-4)) + (j>>(n-4))*(3<<(n-4));
	SB[j<<1] = SA[j<<1]+ SA[(j<<1)+(N>>2)];
	SB[(j<<1)+1] = SA[(j<<1)+1] + SA[(j<<1)+(N>>2)+1];
	SB[(j<<1)+(N>>2)] = SA[j<<1]- SA[(j<<1)+(N>>2)];
	SB[(j<<1)+(N>>2)+1] = SA[(j<<1)+1] - SA[(j<<1)+(N>>2)+1];
	__syncthreads();
	SA[j<<1] = SB[j<<1];
	SA[(j<<1)+1] = SB[(j<<1)+1];
		
	SA[(j<<1)+(N>>2)] =SB[(j<<1)+(N>>2)]*SROT[2*r0]+ SB[(j<<1)+(N>>2)+1]*SROT[2*r0+1];
	SA[(j<<1)+(N>>2)+1] = -SB[(j<<1)+(N>>2)]*SROT[2*r0+1] + SB[(j<<1)+(N>>2)+1]*SROT[2*r0];
	__syncthreads();
	//stage2:
	short ind = (j<<1) + (j>>(n-4))*(1<<(n-3));
	short r1  = ((j%(1<<(n-4)))+(b<<(n-4)))*2;
	SB[ind] = SA[ind]+ SA[ind+(N>>3)];
	SB[ind+1] = SA[ind+1] + SA[ind+(N>>3)+1];
	SB[ind+(N>>3)] = SA[ind]- SA[ind+(N>>3)];
	SB[ind+(N>>3)+1] = SA[ind+1] - SA[ind+(N>>3)+1];
	__syncthreads();
	SA[ind] = SB[ind];
	SA[ind+1] = SB[ind+1];		
	SA[ind+(N>>3)] = SB[ind+(N>>3)]*SROT[2*r1]+ SB[ind+(N>>3)+1]*SROT[2*r1+1];
	SA[ind+(N>>3)+1] = -SB[ind+(N>>3)]*SROT[2*r1+1] + SB[ind+(N>>3)+1]*SROT[2*r1];
	__syncthreads();

	A[j+b*(D)]          = SA[j];
	A[j+b*(D)+(N/2)]   =SA[j+D];
	A[j+b*(D)+N]        =SA[j+2*D];
	A[j+b*(D)+N+(N/2)] =SA[j+3*D] ;

	/*A[j+b*(D)]          = SA[j];
	A[j+b*(D)+32]    =SA[j+D];
	A[j+b*(D)+64]    =SA[j+2*D];
	A[j+b*(D)+96]  =SA[j+3*D] ;*/

	/*A[j+b*(N>>1)] = SA[j];
	A[j+b*(N>>1) + D] = SA[j+D];
	A[j+b*(N>>1) + 2*D] = SA[j+2*D];
	A[j+b*(N>>1) + 3*D] = SA[j+3*D];
	A[j+b*(N>>1)] = SB[j];
	A[j+b*(N>>1) + D] = SB[j+D];
	A[j+b*(N>>1) + 2*D] = SB[j+2*D];
	A[j+b*(N>>1) + 3*D] = SB[j+3*D];*/
	__syncthreads();
       }


	__global__ void kernel1(float* A ,float* ROT)
	{
	__shared__ float SA[M],SB[M],SROT[N];
	short j   = threadIdx.x;
	short b   = blockIdx.x; 
	short i = j>>1;
	short k = j%2;
	
	SA[i] = A[i+b*(N>>1)];
	SA[i+D] = A[i+b*(N>>1) + D];
	SA[i+2*D] = A[i+b*(N>>1) + 2*D];
	SA[i+3*D] = A[i+b*(N>>1) + 3*D];

	SROT[j] = ROT[j];
	SROT[j+blockDim.x] = ROT[j+blockDim.x];
	SROT[j+2*blockDim.x] = ROT[j+2*blockDim.x];
	SROT[j+3*blockDim.x] = ROT[j+3*blockDim.x];

	__syncthreads();
	short ind0 = i<<1;
	short ind1 = (i<<1) + (N>>2);  
	short ind2 = i<<2;
   	short sign =(-2)*k+1;

	for(short s= 3; s<= n; s++)
	  {
		SB[ind2+k] = SA[ind0+k] + SA[ind1+k];
        	SB[ind2+2+k] = SA[ind0+k] - SA[ind1+k];
		__syncthreads();
        	short r0 = (i/(1<<(s-3)))*(1<<(s-1));
        	SA[ind2+k] = SB[ind2+k];
        	SA[ind2+2+k] = sign*SB[ind2+2]*SROT[2*r0+k] + SB[ind2+3]*SROT[2*r0+(!k)];
		__syncthreads();
	
	    }
	A[j+b*(N>>1)] = SA[j];
	A[j+b*(N>>1) + blockDim.x] = SA[j+blockDim.x];
	//A[j+b*(N>>1) + 2*D] = SA[j+2*D];
	//A[j+b*(N>>1) + 3*D] = SA[j+3*D];
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

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	  
          int memsize= 2*N * sizeof(float); 
	   int rotsize = N* sizeof(float);
	   //count = 0;

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
	   
	   //cudaMemcpy(&count, &counth , sizeof(int),  cudaMemcpyHostToDevice);

           //__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(4,1);
           dim3 blockDim0(D,1);
	   dim3 blockDim1(2*D,1);

	   cudaEventRecord(start, 0);
	   cudaMemcpy(ROTd, ROT, rotsize,  cudaMemcpyHostToDevice);
	   kernel0<<<gridDim , blockDim0>>>(Ad,ROTd);
           //cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   //cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);
		
            /*printf("The  outputs of kernel1 are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); */
	    
	    kernel1<<<gridDim , blockDim1>>>(Ad,ROTd);
	    cudaEventRecord(stop, 0);
	   cudaEventSynchronize(stop);

	    cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	    printf("The  outputs of kernel2 are: \n");
            /*for (int k=0; k< N; k++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*k,A[2*k],2*k+1,A[2*k+1]); */
	cudaEventElapsedTime(&time, start, stop);
       	    printf ("Time for the kernel: %f us\n", time*1000.0);

     }
	
