#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

// Utility function for checking CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Activation functions and their derivatives
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// CUDA kernel for element-wise addition
__global__ void matrix_add(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for applying activation function
__global__ void apply_activation(float* input, float* output, int size, bool use_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = use_relu ? relu(input[idx]) : sigmoid(input[idx]);
    }
}

// Structure to hold layer information
struct Layer {
    int input_size;
    int output_size;
    float* weights;
    float* bias;
    float* output;
    bool use_relu;
};

// Neural Network class
class NeuralNetwork {
private:
    int num_layers;
    Layer* layers;
    float* input;
    float* output;

public:
    NeuralNetwork(int* layer_sizes, int num_layers, bool* use_relu) {
        this->num_layers = num_layers - 1;
        layers = new Layer[this->num_layers];

        for (int i = 0; i < this->num_layers; ++i) {
            layers[i].input_size = layer_sizes[i];
            layers[i].output_size = layer_sizes[i + 1];
            layers[i].use_relu = use_relu[i];

            int weight_size = layers[i].input_size * layers[i].output_size;
            int bias_size = layers[i].output_size;

            CHECK_CUDA_ERROR(cudaMalloc(&layers[i].weights, weight_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&layers[i].bias, bias_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&layers[i].output, bias_size * sizeof(float)));

            // Initialize weights and biases (you may want to use cuRAND for better initialization)
            float* h_weights = new float[weight_size];
            float* h_bias = new float[bias_size];

            for (int j = 0; j < weight_size; ++j) {
                h_weights[j] = 0.01f * (float)rand() / RAND_MAX;
            }
            for (int j = 0; j < bias_size; ++j) {
                h_bias[j] = 0.0f;
            }

            CHECK_CUDA_ERROR(cudaMemcpy(layers[i].weights, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(layers[i].bias, h_bias, bias_size * sizeof(float), cudaMemcpyHostToDevice));

            delete[] h_weights;
            delete[] h_bias;
        }

        CHECK_CUDA_ERROR(cudaMalloc(&input, layer_sizes[0] * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&output, layer_sizes[num_layers - 1] * sizeof(float)));
    }

    ~NeuralNetwork() {
        for (int i = 0; i < num_layers; ++i) {
            CHECK_CUDA_ERROR(cudaFree(layers[i].weights));
            CHECK_CUDA_ERROR(cudaFree(layers[i].bias));
            CHECK_CUDA_ERROR(cudaFree(layers[i].output));
        }
        delete[] layers;
        CHECK_CUDA_ERROR(cudaFree(input));
        CHECK_CUDA_ERROR(cudaFree(output));
    }

    void forward(float* h_input) {
        CHECK_CUDA_ERROR(cudaMemcpy(input, h_input, layers[0].input_size * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block_size(256);
        dim3 grid_size((layers[0].output_size + block_size.x - 1) / block_size.x);

        for (int i = 0; i < num_layers; ++i) {
            matrix_multiply<<<grid_size, block_size>>>(
                i == 0 ? input : layers[i-1].output,
                layers[i].weights,
                layers[i].output,
                1, layers[i].input_size, layers[i].output_size
            );

            matrix_add<<<grid_size, block_size>>>(
                layers[i].output,
                layers[i].bias,
                layers[i].output,
                layers[i].output_size
            );

            apply_activation<<<grid_size, block_size>>>(
                layers[i].output,
                layers[i].output,
                layers[i].output_size,
                layers[i].use_relu
            );
        }

        CHECK_CUDA_ERROR(cudaMemcpy(output, layers[num_layers-1].output, layers[num_layers-1].output_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Note: Backward pass and training functions are not implemented in this example

    // Add this getter method
    float* getOutput() const {
        return output;
    }
};

// Example usage
int main() {
    int layer_sizes[] = {2, 4, 3, 1};
    bool use_relu[] = {true, true, false};
    NeuralNetwork nn(layer_sizes, 4, use_relu);

    float input[] = {0.5f, 0.8f};
    float* h_output = new float[1];

    nn.forward(input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, nn.getOutput(), sizeof(float), cudaMemcpyDeviceToHost));

    printf("Output: %f\n", h_output[0]);

    delete[] h_output;
    return 0;
}
