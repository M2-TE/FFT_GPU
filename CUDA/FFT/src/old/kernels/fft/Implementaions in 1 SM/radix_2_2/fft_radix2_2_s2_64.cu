	#define N 64
	#define M 2*N
	#define D 2*N   
       __global__  void fft(float* A , float* ROT ) 
       { 
	__shared__ float SA[M],SB[M],SROT[M];
	short j   = threadIdx.x;
	short n = logf(N)/logf(4);
	short n1 = logf(N)/logf(2);

	SROT[j] = ROT[j];

	SA[j] = A[j];
	__syncthreads();
	
	short ind0,ind1,ind2,r1,r4,sign1,sign2,index1,index2,index3,index4,index5,index6;
	short i = j/2;
	short k = j%2;
	short l = i%4;
	short h = i/2;
	short m = i%2;
	short g = h/2;
	short u = h%2;
	short w = m*u;
	
	//for iteration1:
		short s0 = 1;
		short q0 = (1<<2*(n-s0));
		short p0 = M/(1<<(2*s0));
		short ind_tmp0 = 2*(g+(g/q0)*q0*3);
			
		ind0 = ind_tmp0 + l*p0;
		short sign0 = k*(-2)+1;

		short sign3 = -2*((!k)*w)+1;
		short sign4 = -2*(k*w)+1;
		short x = w*sign4;

		//stage1:
		sign1= u*(-2)+1;
		index1 = ind0+sign1*2*p0;

		SB[ind0+k+x] = sign3*(SA[index1+k] + sign1*SA[ind0+k]);
	
		//stage2:
		sign2= m*(-2)+1;
		s0 = 2;
		r1 = ((((ind0>>(n1-s0+1))%2)<<1)+((ind0>>(n1-s0+2))))*((ind0>>1)%(1<<(n1-s0)));

		//r1 = (ind0/64)*((ind0/2)%16) + m*((ind0/2)%16)*2; 

		index2 = ind0+sign2*p0;

		SA[ind0+k] = SB[index2+k] + sign2*SB[ind0+k];
		
		SB[ind0+k] = sign0*SA[ind0]*SROT[2*r1+k] + SA[ind0+1]*SROT[2*r1+(!k)];
		//__syncthreads();

	//for iteration2:

		short s1 = 2;
		short q1 = (1<<2*(n-s1));
		short p1 = M/(1<<(2*s1));
		short ind_tmp1 = 2*(g+(g/q1)*q1*3);
				
		ind1 = ind_tmp1 + l*p1;

		//stage1:

		index3 = ind1+sign1*2*p1;

		SA[ind1+k+x] = sign3*(SB[index3+k] + sign1*SB[ind1+k]);
				
		//stage2:
		s0 = 4;
		r4=(1<<(s0-2))*((((ind1>>(n1-s0+1))%2)<<1)+((ind1>>(n1-s0+2)))%2)*((ind1>>1)%(1<<(n1-s0)));
		
		//r4 = (((ind1/16)%2)*4 + m*8*((ind1/8)%2))*((ind1/2)%4);
		index4 = ind1+sign2*p1;

		SB[ind1+k] = SA[index4+k] + sign2*SA[ind1+k];
		
		SA[ind1+k] = sign0*SB[ind1]*SROT[2*r4+k] + SB[ind1+1]*SROT[2*r4+(!k)];
		//__syncthreads();
	//for iteration3:
	
		short s2 = 3;
		short q2 = (1<<2*(n-s2));
		short p2 = M/(1<<(2*s2));
		short ind_tmp2 = 2*(g+(g/q2)*q2*3);

		ind2 = ind_tmp2 + l*p2;
		
		//stage1&2:
		index5 = ind2+sign1*2*p2;
		index6 = ind2+sign2*p2;
		
		SB[ind2+k+x] = sign3*(SA[index5+k] + sign1*SA[ind2+k]);
		SA[ind2+k] = SB[index6+k] + sign2*SB[ind2+k];
		
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
		



		

		
