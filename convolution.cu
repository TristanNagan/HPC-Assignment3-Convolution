#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define TILE_SIZE 32
#define kernel_size 11
#define kernel_rad (kernel_size / 2)
__constant__ float devKernel[kernel_size*kernel_size];
texture<float, 2, cudaReadModeElementType> tex;

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void serial_convolution(float *input, float *output, float *kernel, int height, int width){
    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            output[y*width + x] = 0.0f;
            for(int kx = -kernel_rad; kx < kernel_rad + 1; kx++){
                for(int ky = -kernel_rad; ky < kernel_rad + 1; ky++){
                    int ki = (ky + kernel_rad)*kernel_size + (kx + kernel_rad);
                    int ii = (y + ky)*width + (x + kx);
                    if((y + ky) >= 0 && (y + ky) < height && (x + kx) >= 0 && (x + kx) < width){
                        output[y*width + x] += kernel[ki]*input[ii];
                    }else{
                        output[y*width + x] += kernel[ki]*0.0f;
                    }
                }
            }
        }
    }
}

__global__ void global_convolution(float *input, float *output, float *kernel, int height, int width){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    if(x >= 0  && x < width && y >= 0 && y < height){
        output[y*width + x] = 0.0f;
        for(int kx = -kernel_rad; kx < kernel_rad + 1; kx++){
            for(int ky = -kernel_rad; ky < kernel_rad + 1 ; ky++){
                unsigned int ki = (ky + kernel_rad)*kernel_size + (kx + kernel_rad);
                unsigned int ii = (y + ky)*width + (x + kx);
                if((y + ky) >= 0 && (y + ky) < height && (x + kx) >= 0 && (x + kx) < width){
                    output[y*width + x] += kernel[ki]*input[ii];
                }else{
                    output[y*width + x] += kernel[ki]*0.0f;
                }
            }
        }
    }
}

__global__ void global_constant_convolution(float *input, float *output, const float *__restrict__ kernel, int height, int width){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    if(x >= 0  && x < width && y >= 0 && y < height){
        output[y*width + x] = 0.0f;
        for(int kx = -kernel_rad; kx < kernel_rad + 1; kx++){
            for(int ky = -kernel_rad; ky < kernel_rad + 1 ; ky++){
                unsigned int ki = (ky + kernel_rad)*kernel_size + (kx + kernel_rad);
                unsigned int ii = (y + ky)*width + (x + kx);
                if((y + ky) >= 0 && (y + ky) < height && (x + kx) >= 0 && (x + kx) < width){
                    output[y*width + x] += devKernel[ki]*input[ii];
                }else{
                    output[y*width + x] += devKernel[ki]*0.0f;
                }
            }
        }
    }
}

__global__ void shared_convolution(float *input, float *output, float *kernel, int height, int width){
    __shared__ float tile[TILE_SIZE + kernel_size -1][TILE_SIZE + kernel_size -1];

    int x_i = blockDim.x*blockIdx.x + threadIdx.x;
    int y_i = blockDim.y*blockIdx.y + threadIdx.y;

    const int t_loc = y_i*width + x_i;

    int x, y;

    x = x_i - kernel_rad;
    y = y_i - kernel_rad;
    if(x < 0 || y < 0){
        tile[threadIdx.x][threadIdx.y] = 0.0f;
    }else{
        tile[threadIdx.x][threadIdx.y] = input[t_loc - kernel_rad - kernel_rad*width];
    }

    x = x_i + kernel_rad;
    y = y_i - kernel_rad;
    if(x > width - 1 || y < 0){
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y] = 0.0f;
    }else{
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y] = input[t_loc + kernel_rad - kernel_rad*width];
    }

    x = x_i - kernel_rad;
    y = y_i + kernel_rad;
    if(x < 0 || y > height - 1){
        tile[threadIdx.x][threadIdx.y + 2*kernel_rad] = 0.0f;
    }else{
        tile[threadIdx.x][threadIdx.y + 2*kernel_rad] = input[t_loc - kernel_rad + kernel_rad*width];
    }

    x = x_i + kernel_rad;
    y = y_i + kernel_rad;
    if(x > width - 1 || y > height - 1){
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y + 2*kernel_rad] = 0.0f;
    }else{
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y + 2*kernel_rad] = input[t_loc + kernel_rad + kernel_rad*width];
    }
    __syncthreads();

    float sum = 0.0f;
    x = kernel_rad + threadIdx.x;
    y = kernel_rad + threadIdx.y;
    for(int kx = -kernel_rad; kx < kernel_rad + 1; kx++){
        for(int ky = -kernel_rad; ky < kernel_rad + 1; ky++){
            sum += tile[x + kx][y + ky]*kernel[(ky + kernel_rad)*kernel_size + (kx + kernel_rad)];
        }
    }
    output[t_loc] = sum;
}

__global__ void shared_constant_convolution(float *input, float *output, const float *__restrict__ kernel, int height, int width){
    __shared__ float tile[TILE_SIZE + kernel_size -1][TILE_SIZE + kernel_size -1];

    int x_i = blockDim.x*blockIdx.x + threadIdx.x;
    int y_i = blockDim.y*blockIdx.y + threadIdx.y;

    const int t_loc = y_i*width + x_i;

    int x, y;

    x = x_i - kernel_rad;
    y = y_i - kernel_rad;
    if(x < 0 || y < 0){
        tile[threadIdx.x][threadIdx.y] = 0.0f;
    }else{
        tile[threadIdx.x][threadIdx.y] = input[t_loc - kernel_rad - kernel_rad*width];
    }

    x = x_i + kernel_rad;
    y = y_i - kernel_rad;
    if(x > width - 1 || y < 0){
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y] = 0.0f;
    }else{
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y] = input[t_loc + kernel_rad - kernel_rad*width];
    }

    x = x_i - kernel_rad;
    y = y_i + kernel_rad;
    if(x < 0 || y > height - 1){
        tile[threadIdx.x][threadIdx.y + 2*kernel_rad] = 0.0f;
    }else{
        tile[threadIdx.x][threadIdx.y + 2*kernel_rad] = input[t_loc - kernel_rad + kernel_rad*width];
    }

    x = x_i + kernel_rad;
    y = y_i + kernel_rad;
    if(x > width - 1 || y > height - 1){
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y + 2*kernel_rad] = 0.0f;
    }else{
        tile[threadIdx.x + 2*kernel_rad][threadIdx.y + 2*kernel_rad] = input[t_loc + kernel_rad + kernel_rad*width];
    }
    __syncthreads();

    float sum = 0.0f;
    x = kernel_rad + threadIdx.x;
    y = kernel_rad + threadIdx.y;
    for(int kx = -kernel_rad; kx < kernel_rad + 1; kx++){
        for(int ky = -kernel_rad; ky < kernel_rad + 1; ky++){
            sum += tile[x + kx][y + ky]*devKernel[(ky + kernel_rad)*kernel_size + (kx + kernel_rad)];
        }
    }
    output[t_loc] = sum;
}

__global__ void texture_convolution(float *output, float *kernel, int height, int width){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    output[y*width + x] = 0.0f;
    for(int kx = -kernel_rad; kx < kernel_rad + 1 ; kx++){
        for(int ky = -kernel_rad; ky < kernel_rad + 1; ky++){
            int ki = (ky + kernel_rad)*kernel_size + (kx + kernel_rad);
            output[y*width + x] += kernel[ki]*tex2D(tex, x + kx, y + ky);
        }
    }
}

int main(){
    printf("image,width,height,tile_size,kernel_size,serial_time,global_time,global_constant_time,shared_time,shared_constant_time,texture_time,\
serial_tp,global_tp,global_constant_tp,shared_tp,shared_constant_tp,texture_tp\n");
//================================================================== Define Kernel =============================================================================
    /*float kernel[3*3] = {(float)1/9, (float)1/9, (float)1/9, 
                          (float)1/9, (float)1/9, (float)1/9,
                          (float)1/9, (float)1/9, (float)1/9
                          };*/
    /*float kernel[5*5] = {(float)1/25, (float)1/25, (float)1/25, (float)1/25, (float)1/25,
                          (float)1/25, (float)1/25, (float)1/25, (float)1/25, (float)1/25,
                          (float)1/25, (float)1/25, (float)1/25, (float)1/25, (float)1/25,
                          (float)1/25, (float)1/25, (float)1/25, (float)1/25, (float)1/25,
                          (float)1/25, (float)1/25, (float)1/25, (float)1/25, (float)1/25};*/
    /*float kernel[7*7] = {(float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49,
                         (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49, (float)1/49};*/
    /*float kernel[9*9] = {(float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81,
                         (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81, (float)1/81};*/
    float kernel[11*11] = {(float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121,
                           (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121, (float)1/121,
                                                                                    (float)1/121, (float)1/121};
    cudaMemcpyToSymbol(devKernel, kernel, kernel_size*kernel_size*sizeof(float));
//==============================================================================================================================================================

//================================================================ Define Kernel Parameters ====================================================================
            dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
            dim3 dimGrid(1, 1, 1);
//==============================================================================================================================================================

//============================================================== Loop Through the images =======================================================================
    char ref[8][32] = {"lena_bw.pgm", "image21.pgm", "man.pgm", "mandrill.pgm", "ref_rotated.pgm", "lion.pgm", "road.pgm", "winter.pgm"};
    char images[30][32] = {"lena_bw_128.pgm", "lena_bw_256.pgm", "lena_bw_512.pgm", "image21_128.pgm", "image21_256.pgm", "image21_512.pgm",
                          "man_128.pgm", "man_256.pgm", "man_512.pgm", "mandrill_128.pgm", "mandrill_256.pgm", "mandrill_512.pgm",
                          "ref_rotated_128.pgm", "ref_rotated_256.pgm", "ref_rotated_512.pgm", "lion_128.pgm", "lion_256.pgm", "lion_512.pgm",
                          "lion_1024.pgm", "lion_2048.pgm", "road_128.pgm", "road_256.pgm", "road_512.pgm", "road_1024.pgm", "road_2048.pgm",
                          "winter_128.pgm", "winter_256.pgm", "winter_512.pgm", "winter_1024.pgm", "winter_2048.pgm"};
    int refnum = -1;
    int prev_width = 10000;
    int iter = 10;
    for(int i = 0; i < 30; i++){
//==============================================================================================================================================================

//================================================================ Setup Time Variables ========================================================================
        float serial_time = 0;
        float global_time = 0;
        float global_constant_time = 0;
        float shared_time = 0;
        float shared_constant_time = 0;
        float texture_time = 0;
        float time = 0;
        cudaEvent_t launch_begin_seq, launch_end_seq;
//==============================================================================================================================================================

//=================================================================== Load Images ==============================================================================
        float *im = NULL;
        char outputFilename[1024];
        char details[64];
        unsigned int width, height;
        char *imagePath = sdkFindFilePath(images[i], 0);

        if (imagePath == NULL){
            printf("Unable to source image file: %s\n", images[i]);
            exit(EXIT_FAILURE);
        }
        sdkLoadPGM(imagePath, &im, &width, &height);
//==============================================================================================================================================================

//========================================================= Declare and set Host and Device Memory =============================================================
        float *d_im;
        cudaMalloc((void**)&d_im, width*height*sizeof(float));
        cudaMemset(d_im, 0, width*height*sizeof(float));
        cudaMemcpy(d_im, im, width*height*sizeof(float), cudaMemcpyHostToDevice);

        float *h_conv;
        h_conv = (float*)malloc(width*height*sizeof(float));
        memset(h_conv, 0, width*height*sizeof(float));

        float *d_conv;
        cudaMalloc((void**)&d_conv, width*height*sizeof(float));
        cudaMemset(d_conv, 0, width*height*sizeof(float));

        float *d_ker;
        cudaMalloc((void**)&d_ker, kernel_size*kernel_size*sizeof(float));
        cudaMemset(d_ker, 0, kernel_size*kernel_size*sizeof(float));
        cudaMemcpy(d_ker, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
//==============================================================================================================================================================

//============================================================ Define Texture Parameters =======================================================================
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaArray *cuda_im;
        checkCudaErrors(cudaMallocArray(&cuda_im, &channelDesc, width, height));
        checkCudaErrors(cudaMemcpyToArray(cuda_im, 0, 0, im, width*height*sizeof(float), cudaMemcpyHostToDevice));

        tex.normalized = false;
        tex.addressMode[0] = cudaAddressModeBorder;
        tex.addressMode[1] = cudaAddressModeBorder;
        tex.filterMode = cudaFilterModePoint;

        checkCudaErrors(cudaBindTextureToArray(tex, cuda_im, channelDesc));
//==============================================================================================================================================================

//==============================================================================================================================================================
        dimGrid.x = width/dimBlock.x;
        dimGrid.y = height/dimBlock.y;
//==============================================================================================================================================================

//========================================================== Repeat Computations for Averaging =================================================================
        for(int j = 0; j < iter; j++){
//==============================================================================================================================================================

//================================================================== Serial Convolution ========================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            serial_convolution(&im[0], &h_conv[0], &kernel[0], height, width);
            cudaEventRecord(launch_end_seq,0);

            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            serial_time += time;
            sprintf(details, "_serial_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================

//================================================================== Global Convolution ========================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            global_convolution<<<dimGrid, dimBlock>>>(d_im, d_conv, d_ker, height, width);
            cudaEventRecord(launch_end_seq,0);

            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            global_time += time;

            cudaMemcpy(h_conv, d_conv, width*height*sizeof(float), cudaMemcpyDeviceToHost);
            checkCUDAError("global_convolution");

            sprintf(details, "_global_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
            cudaMemset(d_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================

//============================================================= Global Constant Convolution ====================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            global_constant_convolution<<<dimGrid, dimBlock>>>(d_im, d_conv, devKernel, height, width);
            cudaEventRecord(launch_end_seq,0);

            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            global_constant_time += time;

            cudaMemcpy(h_conv, d_conv, width*height*sizeof(float), cudaMemcpyDeviceToHost);

            checkCUDAError("global_constant_convolution");

            sprintf(details, "_g_constant_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
            cudaMemset(d_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================

//================================================================ Shared Convolution ==========================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            shared_convolution<<<dimGrid, dimBlock>>>(d_im, d_conv, d_ker, height, width);
            cudaEventRecord(launch_end_seq,0);

            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            shared_time += time;

            cudaMemcpy(h_conv, d_conv, width*height*sizeof(float), cudaMemcpyDeviceToHost);

            checkCUDAError("shared_convolution");

            sprintf(details, "_shared_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
            cudaMemset(d_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================

//============================================================ Shared Constant Convolution =====================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            shared_constant_convolution<<<dimGrid, dimBlock>>>(d_im, d_conv, devKernel, height, width);
            cudaEventRecord(launch_end_seq,0);

            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            shared_constant_time += time;

            cudaMemcpy(h_conv, d_conv, width*height*sizeof(float), cudaMemcpyDeviceToHost);

            checkCUDAError("shared_constant_convolution");

            sprintf(details, "_s_constant_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
            cudaMemset(d_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================

//============================================================== Texture Convolution ===========================================================================
            cudaEventCreate(&launch_begin_seq);
            cudaEventCreate(&launch_end_seq);  

            cudaEventRecord(launch_begin_seq,0);
            texture_convolution<<<dimGrid, dimBlock>>>(d_conv, d_ker, height, width);
            cudaEventRecord(launch_end_seq,0);

            checkCudaErrors(cudaDeviceSynchronize());
            cudaEventSynchronize(launch_end_seq);
            
            cudaEventElapsedTime(&time, launch_begin_seq, launch_end_seq);
            texture_time += time;

            cudaMemcpy(h_conv, d_conv, width*height*sizeof(float), cudaMemcpyDeviceToHost);

            checkCUDAError("texture_convolution");

            sprintf(details, "_texture_k%d_ts%d.pgm", kernel_size, TILE_SIZE);
            strcpy(outputFilename, imagePath);
            strcpy(outputFilename + strlen(imagePath) - 4, details);
            sdkSavePGM(outputFilename, h_conv, width, height);
            memset(h_conv, 0, width*height*sizeof(float));
            cudaMemset(d_conv, 0, width*height*sizeof(float));
//==============================================================================================================================================================
        }
//=========================================================== Increment Image Reference ========================================================================
        if(prev_width > width){
            refnum++;
        }
        prev_width = width;
//==============================================================================================================================================================

//================================================================= Print Results ==============================================================================
            printf("%s,%i,%i,%i,%i,%f,%f,%f,%f,%f,%f\n", ref[refnum], width, height, TILE_SIZE, kernel_size, serial_time/iter, global_time/iter,
                   global_constant_time/iter, shared_time/iter, shared_constant_time/iter, texture_time/iter);
//==============================================================================================================================================================

//============================================================ Free and Unbind Memory ==========================================================================
        free(im);
        free(h_conv);
        cudaUnbindTexture(tex);
        cudaFree(d_im);
        cudaFreeArray(cuda_im);
        cudaFree(d_conv);
        cudaFree(d_ker);
//==============================================================================================================================================================
    }
    return 0;
}
