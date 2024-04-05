#include <wb.h>

__global__ void vecAdd(float* in1, float* in2, float* out, int len) {

    int i;
    for (i = 0; i <= len; i = i + 1){ 
    	out[i] = in1[i] + in2[i];
    }
    //@@ Insert code to implement vector addition here
}

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* hostInput1;
    float* hostInput2;
    float* hostOutput;
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
        (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
        (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**)&deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceInput2, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));
    //@@ Allocate GPU memory here

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    //@@ Copy memory to the GPU here

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    dim3 dimgrid(256, 1, 1);
    dim3 dimblock (256, 1, 1);
    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    vecAdd <<<dimgrid, dimblock >>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
    //@@ Launch the GPU Kernel here

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    //@@ Copy the GPU memory back to the CPU here

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    //@@ Free the GPU memory here

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

