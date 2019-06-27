#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<cuda.h>


#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535989


#define CUDA_CALL(x) do { if(x!= cudaSuccess) {\
   printf("Error at %s:%d -- %s \n", __FILE__,__LINE__, cudaGetErrorString(x)); \
   return EXIT_FAILURE;}} while(0)


__global__ void monteWithGPU( curandState *states, float * estimate) {
	double x, y;
    int toss, number_in_circle=0;
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int seed = id;
    curand_init(1234, seed ,0 ,&states[id]);

	for(toss=0; toss <TRIALS_PER_THREAD; toss++) {
        /*curand_uniform() range 0~1*/
		x = curand_uniform(&states[id])*2 - 1;
        y = curand_uniform(&states[id])*2 - 1;
	    
        if (x*x+y*y <= 1.0f)
            number_in_circle++;
	}	
    estimate[id] = 4.0f *number_in_circle / (TRIALS_PER_THREAD) ;
}

int main(void) {
	double pi_estimate;
	//time variables
	clock_t start, end;
	double cpu_time_used;
    float number_in_circle = 0;
    curandState *devStates;
    float *dev, *host;

	start = clock();

    host = (float*)malloc(sizeof(float)* THREADS * BLOCKS );
    cudaMalloc((void**) &dev, BLOCKS * THREADS * sizeof(float));
    cudaMalloc((void**)&devStates, THREADS * BLOCKS * sizeof(curandState));
    monteWithGPU<<<BLOCKS, THREADS>>>(devStates, dev);
    cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i < BLOCKS * THREADS ; i++){
        number_in_circle+=host[i];
    }

    pi_estimate = number_in_circle/ (BLOCKS * THREADS);
	end = clock();
	cpu_time_used = ((double)(end - start)) /CLOCKS_PER_SEC;
	printf("pi_estimate = %f \n", pi_estimate);
	printf("Elapsed time = %f seconds \n", cpu_time_used);
    return 0;	
}
