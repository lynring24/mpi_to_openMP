#include <stdio.h> 
#include <curand_kernel.h>
#define CURAND_CALL(x) do { \
    if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \ 
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[]) {
    int n = 100; 
    int i; 
    curandGenerator_t gen; 
    float *devData, *hostData; 
    
    /* Allocate n floats on host */ 
    hostData = (float *)calloc(n, sizeof(float)); 
    
    /* Allocate n floats on device */ 
    cudaMalloc(&devData, n*sizeof(float));

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */ 
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234567));
    
    /* Generate n floats on device */ 
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
    /* Copy device memory to host */ 
    cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("hello world\n");    
    /* Show result */ 
    for(i = 0; i < n; i++) { 
        printf("%1.4f ", hostData[i]); }
    printf("\n"); 
    /* Cleanup */ 
    CURAND_CALL(curandDestroyGenerator(gen)); 
    cudaFree(devData); 
    free(hostData); 
    return 0;
}
