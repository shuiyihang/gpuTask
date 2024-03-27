#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#define BLOCK_X 16
#define BLOCK_Y 16

// 仅测试方差为1 的高斯一阶偏导核
#define VARIANCE    1
#define MASK_WIDTH  (6*VARIANCE + 1)
#define GRID_ROWS   1

#define BLOCKSIZE   256

__managed__ int max_val = 0;

enum DIR_T
{
    DIR_X = 0,
    DIR_Y
};

__host__ int cpu_find_max_value(uchar* img,int size)
{
    int max = 0;
    for(int i = 0;i < size;i++)
    {
        if(max < img[i])
        {
            max = img[i];
        }
    }
    return max;
}
// host函数
__host__ float* get_conv_kernel(int theta,DIR_T type)
{
    float ratio_pi = 0.1592f;// 1/(2*pi)
    float ratio_theta = 1/pow(theta,4);

    float *conv_kernel;
    int size = 2 * 3 * theta + 1;
    cudaMallocHost(&conv_kernel,size * size * sizeof(float));
    int x,y;
    float sum = 0.0000f;

    if(conv_kernel == NULL)
    {
        cout << "conv_kernel malloc failed\n"<<endl;
        return NULL;
    }
    if(type == DIR_X)
    {
        for(int i = 0;i < size; i++)
        {
            for(int j = 0;j < size; j++)
            {
                y = -3 * theta + i;
                x = -3 * theta + j;
                conv_kernel[size * i + j] = ratio_pi * ratio_theta * exp(-1 * (x*x + y*y)/(2 * theta * theta))*(-x);
                // cout << x <<" "<< y <<" "<< i <<" "<< j <<" "<< conv_kernel[size * i + j] <<endl;
                sum += conv_kernel[size * i + j];
            }
        }
    }else
    {
        for(int i = 0;i < size; i++)
        {
            for(int j = 0;j < size; j++)
            {
                y = -3 * theta + i;
                x = -3 * theta + j;
                conv_kernel[size * i + j] = ratio_pi * ratio_theta * exp(-1 * (x*x + y*y)/(2 * theta * theta))*(-y);
                sum += conv_kernel[size * i + j];
            }
        }
    }

    // printf("sum:%.5f\n",sum);
    // normalize
    if(fabs(sum - 0.000f) > 0.000001f)
    {
        for(int i = 0;i < size; i++)
        {
            for(int j = 0;j < size; j++)
            {
                conv_kernel[size * i + j] /= sum;
            }
        }
    }
    return conv_kernel;
}

/**
 * .---->x
 * |
 * |
 * y
*/
__global__ void gpu_convolution(uchar* src_img,uchar* output_img,float* conv_kernel,int width,int height)
{
    __shared__ uchar mem_img_data[BLOCK_Y + MASK_WIDTH - 1][BLOCK_X + MASK_WIDTH - 1];
    
    // if(threadIdx.x == 0)
    // {
    //     printf("grid:%d block:%d\n",gridDim.y,blockDim.y);
    // }
    for(int h = 0;h < height;h += (gridDim.y * blockDim.y))
    {
        // copy data to shared mem

        int row = blockDim.y * blockIdx.y + threadIdx.y + h;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        int x_id = threadIdx.x;
        int y_id = threadIdx.y;

        int row_tmp;
        int col_tmp;

        if(y_id == 0)
        {
            if(x_id == 0)
            {
                // 左上角
                for(int i = 0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        row_tmp = row - 3 + i;
                        col_tmp = col - 3 + j;
                        if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                        {
                            mem_img_data[i][j] = 0;
                        }else
                        {
                            mem_img_data[i][j] = src_img[row_tmp*width +col_tmp];
                        }
                    }   
                }
            }else if(x_id == BLOCK_X - 1)
            {
                // 右上角
                for(int i = 0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        row_tmp = row - 3 + i;
                        col_tmp = col + j;
                        if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                        {
                            mem_img_data[i][x_id + 3 + j] = 0;
                        }else
                        {
                            mem_img_data[i][x_id + 3 + j] = src_img[row_tmp*width +col_tmp];
                        }
                    }   
                }
            }else
            {
                // 上面三个 + 本身
                for(int i = 0;i < 4;i++)
                {
                    row_tmp = row - 3 + i;
                    col_tmp = col;
                    if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                    {
                        mem_img_data[i][x_id + 3] = 0;
                    }else
                    {
                        mem_img_data[i][x_id + 3] = src_img[row_tmp*width +col_tmp];
                    }
                }
            }
        }else if(y_id == BLOCK_Y - 1)
        {
            if(x_id == 0)
            {
                // 左下角
                for(int i = 0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        row_tmp = row + i;
                        col_tmp = col - 3 + j;
                        if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                        {
                            mem_img_data[y_id + 3 + i][j] = 0;
                        }else
                        {
                            mem_img_data[y_id + 3 + i][j] = src_img[row_tmp*width +col_tmp];
                        }
                    }   
                }

            }else if(x_id == BLOCK_X - 1)
            {
                // 右下角
                for(int i = 0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        row_tmp = row + i;
                        col_tmp = col + j;
                        if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                        {
                            mem_img_data[y_id + 3 + i][x_id + 3 + j] = 0;
                        }else
                        {
                            mem_img_data[y_id + 3 + i][x_id + 3 + j] = src_img[row_tmp*width +col_tmp];
                        }
                    }   
                }
            }else
            {
                // 下面三个
                for(int i = 0;i < 4;i++)
                {
                    row_tmp = row + i;
                    col_tmp = col;
                    if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                    {
                        mem_img_data[y_id + 3 + i][x_id + 3] = 0;
                    }else
                    {
                        mem_img_data[y_id + 3 + i][x_id + 3] = src_img[row_tmp*width +col_tmp];
                    }
                }
            }
        }else if(x_id == 0)
        {
            // 左边三个
            for(int i = 0;i < 4;i++)
            {
                row_tmp = row;
                col_tmp = col - 3 + i;
                if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                {
                    mem_img_data[y_id + 3][i] = 0;
                }else
                {
                    mem_img_data[y_id + 3][i] = src_img[row_tmp*width +col_tmp];
                }
            }

        }else if(x_id == BLOCK_X - 1)
        {
            // 右边三个
            for(int i = 0;i < 4;i++)
            {
                row_tmp = row;
                col_tmp = col + i;
                if(row_tmp < 0 || row_tmp >= height || col_tmp < 0 || col_tmp >= width)
                {
                    mem_img_data[y_id + 3][x_id + 3 + i] = 0;
                }else
                {
                    mem_img_data[y_id + 3][x_id + 3 + i] = src_img[row_tmp*width +col_tmp];
                }
            }
        }else
        {
            if(row < 0 || row >= height || col < 0 || col >= width)
            {
                mem_img_data[x_id + 3][y_id + 3] = 0;
            }else
            {
                mem_img_data[x_id + 3][y_id + 3] = src_img[row*width +col];
            }
        }

        __syncthreads();
        
        float sum = 0.0000f;
        if(col < width && row < height)
        {
            for(int i = 0;i < MASK_WIDTH;i++)
            {
                for(int j = 0;j < MASK_WIDTH;j++)
                {
                    sum += mem_img_data[y_id + i][x_id + j] * conv_kernel[i * MASK_WIDTH + j];
                }
            }
            output_img[width*row + col] = sum;
        }
    }

}



__global__ void gpu_merge_grid_xy(uchar* x_img,uchar* y_img,uchar* xy_img,int width,int height)
{
    for(int h = 0;h < height;h += (GRID_ROWS * BLOCK_Y))
    {
        int row = blockDim.y * blockIdx.y + threadIdx.y + h;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        if(col < width && row < height)
        {
            xy_img[width*row + col] = sqrt(pow(x_img[width*row + col],2) + pow(y_img[width*row + col],2));
        }
    }
}

__global__ void gpu_find_max_val(uchar* xy_img,int* max_val,int length)
{
    __shared__ unsigned char mem_max_block[BLOCKSIZE];// n * blocks
    unsigned char tmp_max = 0;
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < length; i += (gridDim.x * blockDim.x))
    {
        if(xy_img[i] > tmp_max)
        {
            tmp_max = xy_img[i];
        }
    }
    mem_max_block[threadIdx.x] = tmp_max;
    // printf("=====%d====\n",tmp_max);
    __syncthreads();

    for(int len = BLOCKSIZE/2; len >= 32; len >>= 1)
    {
        if(threadIdx.x < len)
        {
            mem_max_block[threadIdx.x] = (mem_max_block[threadIdx.x]>mem_max_block[threadIdx.x + len])?mem_max_block[threadIdx.x]:mem_max_block[threadIdx.x + len];
        }
        __syncthreads();
    }

    int val = mem_max_block[threadIdx.x];
    for(int i = 16;i > 0; i /= 2)
    {
        int tmp_val = __shfl_down_sync(0xffffffff,val,i);
        val = (val>tmp_val)?val:tmp_val;
    }

    if(blockDim.x * blockIdx.x < length)
    {
        
        if(threadIdx.x == 0)
        {
            atomicMax(max_val,val);
        }
    }

}

// 合并
// atomicMax(res,val)
__global__ void gpu_normalize(uchar* xy_img,int width,int height,int max)
{
    for(int h = 0;h < height;h += (GRID_ROWS * BLOCK_Y))
    {
        int row = blockDim.y * blockIdx.y + threadIdx.y + h;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        if(col < width && row < height)
        {
            xy_img[width*row + col] = 255 * xy_img[width*row + col] / max;
        }
    }
}
int main()
{
    Mat origin_img = imread("./input_img.png");// w-959 h-640

    Mat gray_img;
    cvtColor(origin_img,gray_img,COLOR_RGB2GRAY);
    if(gray_img.empty())
    {
        cout << "get gray img failed" << endl;
        return -1;
    }

    int img_data_size = gray_img.rows * gray_img.cols * sizeof(uchar);
    printf("img size:(%d,%d)\n",gray_img.rows, gray_img.cols);
    uchar *dev_gray_img;
    uchar *dev_grad_x_img;
    uchar *dev_grad_y_img;
    uchar *dev_grad_xy;
    uchar *host_grad_xy;

    cudaMalloc(&dev_gray_img, img_data_size);
    if(dev_gray_img == NULL){
        cout << "dev_gray_img malloc failed"<<endl;
        return;
    }
    cudaMalloc(&dev_grad_x_img, img_data_size);
    if(dev_grad_x_img == NULL){
        cout << "dev_grad_x_img malloc failed"<<endl;
        return;
    }
    cudaMalloc(&dev_grad_y_img, img_data_size);
    if(dev_grad_y_img == NULL){
        cout << "dev_grad_y_img malloc failed"<<endl;
        return;
    }
    cudaMalloc(&dev_grad_xy, img_data_size);
    if(dev_grad_xy == NULL){
        cout << "dev_grad_xy malloc failed"<<endl;
        return;
    }
    cudaMallocHost(&host_grad_xy,img_data_size);

    cudaMemcpy(dev_gray_img, gray_img.data,img_data_size,cudaMemcpyHostToDevice);

    float *conv_x_kernel = get_conv_kernel(1,DIR_X);
    float *conv_y_kernel = get_conv_kernel(1,DIR_Y); // host


    unsigned int grid_rows = GRID_ROWS;//(gray_img.rows + BLOCK_Y - 1)/BLOCK_Y/4;
    unsigned int grid_cols = (gray_img.cols + BLOCK_X - 1)/BLOCK_X;
    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(BLOCK_Y,BLOCK_X);


    //计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);	

    // 得到xy方向梯度
    gpu_convolution<<<dimGrid,dimBlock>>>(dev_gray_img,dev_grad_x_img,conv_x_kernel,gray_img.cols,gray_img.rows);
    gpu_convolution<<<dimGrid,dimBlock>>>(dev_gray_img,dev_grad_y_img,conv_y_kernel,gray_img.cols,gray_img.rows);
    cudaDeviceSynchronize();
  


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms.\n", elapsed_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    gpu_merge_grid_xy<<<dimGrid,dimBlock>>>(dev_grad_x_img,dev_grad_y_img,dev_grad_xy,gray_img.cols,gray_img.rows);
    cudaDeviceSynchronize();
    int grid_size = (img_data_size + BLOCKSIZE - 1)/BLOCKSIZE;

    gpu_find_max_val<<<grid_size,BLOCKSIZE>>>(dev_grad_xy,&max_val,img_data_size);
    cudaDeviceSynchronize();
    cout << "max_val: " << max_val << endl;
    // 非极大化抑制
    gpu_normalize<<<dimGrid,dimBlock>>>(dev_grad_xy,gray_img.cols,gray_img.rows,max_val);


    // 输出
    cudaMemcpy(host_grad_xy,dev_grad_xy,img_data_size,cudaMemcpyDeviceToHost);
    Mat save_gray_img(gray_img.rows,gray_img.cols,CV_8U,host_grad_xy);
    imwrite("./output_img.jpg",save_gray_img);


    
    cudaFreeHost(conv_x_kernel);
    cudaFreeHost(conv_y_kernel);
    cudaFree(dev_gray_img);
    cudaFree(dev_grad_x_img);
    cudaFree(dev_grad_y_img);
    cudaFree(dev_grad_xy);
    cudaFreeHost(host_grad_xy);
    return 0;
}