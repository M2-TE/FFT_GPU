	#define N 64
	#define n 6
	#define M N/2
	#define D N/4
	__global__ void kernel0(float* A ,float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SROT[2*N];
	short j   = threadIdx.x;
	short b   = blockIdx.x;  
	//short n = logf(N)/logf(2);

	short i = j>>1;
	short k = j%2;
	//short g = j+b*D;
	
	SA[i] = A[i + b*(N>>3)];
	SA[i+(N>>3)] = A[i + b*(N>>3) + (N>>1)];
	SA[i+2*(N>>3)] = A[i + b*(N>>3) + N];
	SA[i+3*(N>>3)] = A[i + b*(N>>3) + N + (N>>1)];

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
	short ind0 = i<<1;
	short ind1 =(i<<1) + (N>>2);
	short r0 = (j%2)*(j>>(n-3)); //(j%2)*(j>>(n-1))
	short signk = -(k<<1) + 1;
	short signr0 = -(r0<<1) + 1;
	SB[ind0+k*(N>>2)+r0] = signr0*(SA[ind0] + signk*SA[ind1]);
	SB[ind0+k*(N>>2)+(!r0)] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();
	//stage2:
	short ind = (i<<1) + (i>>(n-4))*(1<<(n-3));
	short x = ((j%2)<<1) + (j>>(n-3));
	short r1 = ((x*b)*(N>>4)) + x*((j>>1)%(1<<(n-4)));
	//short r1 = (((g%2)<<1)+((g>>1)%2))*(g>>2);
	SA[ind+k*(N>>3)] = SB[ind] + signk*SB[ind+(N>>3)];
	SA[ind+k*(N>>3)+1] = SB[ind+1] + signk*SB[ind+(N>>3)+1];
	__syncthreads(); 
	SB[ind+k*(N>>3)]   = SA[ind+k*(N>>3)]*SROT[r1<<1] + SA[ind+k*(N>>3)+1]*SROT[(r1<<1)+1];
	SB[ind+k*(N>>3)+1] = -SA[ind+k*(N>>3)]*SROT[(r1<<1)+1] + SA[ind+k*(N>>3)+1]*SROT[r1<<1];
	__syncthreads();
	A[i + b*(N>>3)]          = SB[i];
	A[i + b*(N>>3) + (N>>1)]   =SB[i+(N>>3)];
	A[i + b*(N>>3) + N]        =SB[i+2*(N>>3)];
	A[i + b*(N>>3) + N + (N>>1)] =SB[i+3*(N>>3)];
	
	/*A[i + b*(N>>3)]          = SA[i];
	A[i + b*(N>>3) + (N>>1)]   =SA[i+(N>>3)];
	A[i + b*(N>>3) + N]        =SA[i+2*(N>>3)];
	A[i + b*(N>>3) + N + (N>>1)] =SA[i+3*(N>>3)];*/

	//A[j+b*(N>>1)] = SB[j];
	//A[j+b*(N>>1)+D] = SB[j+D];
      }

	__global__ void kernel1(float* A ,float* ROT) 
       { 
	__shared__ float SA[M],SB[M],SROT[2*N];
	short j   = threadIdx.x;
	short b   = blockIdx.x;  
	//short n = logf(N)/logf(2);

	short i = j>>1;
	short k = j%2;
	//short g = j+b*D;
	
	/*SA[i] = A[i + b*(N>>3)];
	SA[i+(N>>3)] = A[i + b*(N>>3) + (N>>1)];
	SA[i+2*(N>>3)] = A[i + b*(N>>3) + N];
	SA[i+3*(N>>3)] = A[i + b*(N>>3) + N + (N>>1)];*/

	SA[j] = A[j+b*(N>>1)];
	SA[j+D] = A[j+b*(N>>1) + D];
	
	SROT[j] = ROT[j];
	SROT[j+D] = ROT[j+D];
	SROT[j+2*D] = ROT[j+2*D];
	SROT[j+3*D] = ROT[j+3*D];
	SROT[j+4*D] = ROT[j+4*D];
	SROT[j+5*D] = ROT[j+5*D];
	SROT[j+6*D] = ROT[j+6*D];
	SROT[j+7*D] = ROT[j+7*D];
	__syncthreads();
	//stage3:
	short ind0 = i<<1;
	short ind1 =(i<<1) + (N>>2);
	short ind2 = j<<1;
	short r0 = (j%2)*(j>>(n-3)); 
	short signk = -(k<<1) + 1;
	short signr0 = -(r0<<1) + 1;
	SB[ind2+r0] = signr0*(SA[ind0] + signk*SA[ind1]);
	SB[ind2+(!r0)] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();
	//stage4:
	short s = 4;
	short r3 = (1<<(s-2))*(((j%2)<<1) + ((j>>1)%2))*((j%(1<<(n-2)))>>(s-2));
	SA[ind2] = SB[ind0] + signk*SB[ind1];
	SA[ind2+1] = SB[ind0+1] + signk*SB[ind1+1];
	__syncthreads(); 
	SB[ind2]   = SA[ind2]*SROT[r3<<1] + SA[ind2+1]*SROT[(r3<<1)+1];
	SB[ind2+1] = -SA[ind2]*SROT[(r3<<1)+1] + SA[ind2+1]*SROT[r3<<1];
	__syncthreads();
	//stage5:
	SA[ind2+r0] = signr0*(SB[ind0] + signk*SB[ind1]);
	SA[ind2+(!r0)] = SB[ind0+1] + signk*SB[ind1+1];
	__syncthreads();
	//stage6:
	SB[ind2] = SA[ind0] + signk*SA[ind1];
	SB[ind2+1] = SA[ind0+1] + signk*SA[ind1+1];
	__syncthreads();
	A[j+b*(N>>1)] = SA[j];
	A[j+b*(N>>1)+D] = SA[j+D];
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
	   //count = 0;

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
	   //cudaMemcpy(&count, &counth , sizeof(int),  cudaMemcpyHostToDevice);

           //__global__ functions are called:  Func<<< Dg, Db, Ns  >>>(parameter); 
           dim3 gridDim(4,1);
           dim3 blockDim(D,1);
	
	   kernel0<<<gridDim , blockDim>>>(Ad,ROTd);
           cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	   cudaMemcpy(ROT, ROTd, rotsize,  cudaMemcpyDeviceToHost);
		
            printf("The  outputs of kernel1 are: \n");
            for (int l=0; l< N; l++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*l,A[2*l],2*l+1,A[2*l+1]); 
	    
	    kernel1<<<gridDim , blockDim>>>(Ad,ROTd);
	    cudaMemcpy(A, Ad, memsize,  cudaMemcpyDeviceToHost); 
	    printf("The  outputs of kernel2 are: \n");
            for (int k=0; k< N; k++) 
            printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ",2*k,A[2*k],2*k+1,A[2*k+1]); 

     }
	




















