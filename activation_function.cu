#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)


// relu activation 

__global__ void relu(const float* x, float* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_derivative(const float* x, float* dx, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dx[i] = (x[i] > 0.0f ? 1.0f : 0.0f);
}


// Elementwise ops
__global__ void subtract(const float* a, const float* b, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void multiply(const float* a, const float* b, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}


// Matrix multiply
// C[m×p] = A[m×n] × B[n×p]
__global__ void matmul(const float* A, const float* B, float* C,
                       int m, int n, int p)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = m * p;
    if (idx >= size) return;

    int r = idx / p;
    int c = idx % p;

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += A[r * n + k] * B[k * p + c];

    C[idx] = sum;
}

// C += A^T × B
__global__ void matmul_AT_B_add(const float* A, const float* B, float* C,
                                int m, int n, int p)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = n * p;
    if (idx >= size) return;

    int r = idx / p;  // n
    int c = idx % p;  // p

    float sum = 0.0f;
    for (int k = 0; k < m; k++)
        sum += A[k * n + r] * B[k * p + c];

    C[idx] += sum;
}


// Utilities

void launch1D(dim3 &grid, dim3 &block, int n) {
    block = dim3(256);
    grid  = dim3((n + block.x - 1) / block.x);
}



// Main MLP Training Loop

int main() {

    // Network sizes
    const int N = 4;      // samples
    const int D = 4;      // input dims
    const int H = 8;      // hidden units
    const int O = 1;      // output units
    const float lr = 0.01f;

    // Host data

    float h_X[N*D] = {
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        6.2, 3.4, 5.4, 2.3,
        5.9, 3.0, 5.1, 1.8
    };

    float h_y[N] = {0,0,1,1};

    // Random init weights
    float h_W0[D*H];
    float h_W1[H*O];
    for (int i = 0; i < D*H; i++) h_W0[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < H*O; i++) h_W1[i] = 0.1f * ((float)rand()/RAND_MAX - 0.5f);


    // Device buffers
    float *X, *y, *W0, *W1;
    float *L1, *L1_relu, *L1_d;
    float *pred, *pred_d, *buffer;

    CHECK(cudaMalloc(&X, N*D*sizeof(float)));
    CHECK(cudaMalloc(&y, N*sizeof(float)));
    CHECK(cudaMalloc(&W0, D*H*sizeof(float)));
    CHECK(cudaMalloc(&W1, H*O*sizeof(float)));

    CHECK(cudaMalloc(&L1, N*H*sizeof(float)));
    CHECK(cudaMalloc(&L1_relu, N*H*sizeof(float)));
    CHECK(cudaMalloc(&L1_d, N*H*sizeof(float)));

    CHECK(cudaMalloc(&pred, N*O*sizeof(float)));
    CHECK(cudaMalloc(&pred_d, N*O*sizeof(float)));
    CHECK(cudaMalloc(&buffer, N*H*sizeof(float)));

    CHECK(cudaMemcpy(X,  h_X, N*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y,  h_y, N*sizeof(float),   cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(W0, h_W0, D*H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(W1, h_W1, H*O*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid, block;

    // TRAIN
    for (int iter = 0; iter < 2000; iter++) {

        // Forward: L1 = X W0
        launch1D(grid, block, N*H);
        matmul<<<grid, block>>>(X, W0, L1, N, D, H);

        // ReLU activation
        relu<<<grid, block>>>(L1, L1_relu, N*H);


        // Forward: pred = L1 W1
        launch1D(grid, block, N*O);
        matmul<<<grid, block>>>(L1_relu, W1, pred, N, H, O);

        // Backprop output d(pred)
        // pred_d = pred - y
        subtract<<<grid, block>>>(pred, y, pred_d, N*O);
        
        // Update W1: W1 -= lr * (L1_reluᵀ × pred_d)
        CHECK(cudaMemset(W1, 0, H*O*sizeof(float)));
        launch1D(grid, block, H*O);
        matmul_AT_B_add<<<grid, block>>>(L1_relu, pred_d, W1, N, H, O);

        // Scale by learning rate
        launch1D(grid, block, H*O);
        multiply<<<grid, block>>>(W1, W1, W1, H*O); 

        // Hidden layer delta: L1_d = (pred_d × W1ᵀ) ⊙ ReLU'(L1)
        launch1D(grid, block, N*H);
        matmul<<<grid, block>>>(pred_d, W1, buffer, N, O, H); // pred_d * W1ᵀ

        relu_derivative<<<grid, block>>>(L1, L1_d, N*H);
        multiply<<<grid, block>>>(buffer, L1_d, L1_d, N*H);

        // Update W0: W0 -= lr * (Xᵀ × L1_d)
        CHECK(cudaMemset(W0, 0, D*H*sizeof(float)));
        launch1D(grid, block, D*H);
        matmul_AT_B_add<<<grid, block>>>(X, L1_d, W0, N, D, H);

    }

    // Read back prediction
    float h_pred[N];
    CHECK(cudaMemcpy(h_pred, pred, N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("Predictions:\n");
    for (int i = 0; i < N; i++)
        printf("Sample %d: %.3f (truth = %.1f)\n", i, h_pred[i], h_y[i]);

    return 0;
}
