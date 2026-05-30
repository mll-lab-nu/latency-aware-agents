#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr unsigned int PCG_DEFAULT_MULTIPLIER_32 = 747796405U;
constexpr unsigned int PCG_DEFAULT_INCREMENT_32 = 2891336453U;

__global__ void demon_attack_noop_kernel(unsigned char* out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = out[idx];
    }
}

__device__ __forceinline__ unsigned int pcg_sample(unsigned int* state) {
    const unsigned int oldstate = *state;
    *state = oldstate * PCG_DEFAULT_MULTIPLIER_32 + PCG_DEFAULT_INCREMENT_32;
    const unsigned int word = ((oldstate >> ((oldstate >> 28U) + 4U)) ^ oldstate) * 277803737U;
    return (word >> 22U) ^ word;
}

__device__ __forceinline__ int bcd_byte_score(unsigned char value, int multiplier) {
    const int right = static_cast<int>(value & 0x0F);
    const int left = static_cast<int>(value >> 4);
    return multiplier * ((10 * left) + right);
}

__global__ void demon_attack_decode_ram_kernel(
    const unsigned char* __restrict__ ram,
    const int* __restrict__ previous_score,
    int* __restrict__ reward,
    int* __restrict__ score,
    int* __restrict__ lives,
    bool* __restrict__ terminal,
    int64_t batch_size) {
    const int64_t env = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }
    const unsigned char* env_ram = ram + env * 128;
    int current_score = (
        bcd_byte_score(env_ram[0x05], 10000) +
        bcd_byte_score(env_ram[0x03], 100) +
        bcd_byte_score(env_ram[0x01], 1));
    if (env_ram[0x01] == 0xAB && env_ram[0x03] == 0xCD && env_ram[0x05] == 0xEA) {
        current_score = 0;
    }

    const int lives_displayed = static_cast<int>(env_ram[0x72]);
    const bool is_terminal = lives_displayed == 0 && env_ram[0x71] == 0xBD;
    score[env] = current_score;
    reward[env] = current_score - previous_score[env];
    terminal[env] = is_terminal;
    lives[env] = is_terminal ? 0 : lives_displayed + 1;
}

__global__ void demon_attack_reset_indices_kernel(
    const int64_t* __restrict__ indices,
    unsigned char* __restrict__ states,
    unsigned char* __restrict__ frame_states,
    unsigned char* __restrict__ ram,
    const unsigned char* __restrict__ cached_states,
    const unsigned char* __restrict__ cached_frame_states,
    const unsigned char* __restrict__ cached_ram,
    unsigned int* __restrict__ rand_states,
    unsigned int* __restrict__ cache_index,
    int64_t state_size,
    int64_t frame_state_size,
    int64_t ram_size,
    int64_t noop_reset_steps,
    int64_t count) {
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }
    const int64_t env = indices[item];
    const unsigned int sampled = pcg_sample(rand_states + env);
    const int64_t reset_slot = static_cast<int64_t>(sampled % static_cast<unsigned int>(noop_reset_steps));
    cache_index[env] = static_cast<unsigned int>(reset_slot);

    unsigned char* state_dst = states + env * state_size;
    const unsigned char* state_src = cached_states + reset_slot * state_size;
    for (int64_t offset = 0; offset < state_size; ++offset) {
        state_dst[offset] = state_src[offset];
    }

    unsigned char* frame_state_dst = frame_states + env * frame_state_size;
    const unsigned char* frame_state_src = cached_frame_states + reset_slot * frame_state_size;
    for (int64_t offset = 0; offset < frame_state_size; ++offset) {
        frame_state_dst[offset] = frame_state_src[offset];
    }

    unsigned char* ram_dst = ram + env * ram_size;
    const unsigned char* ram_src = cached_ram + reset_slot * ram_size;
    for (int64_t offset = 0; offset < ram_size; ++offset) {
        ram_dst[offset] = ram_src[offset];
    }
}

}  // namespace

void demon_attack_noop_cuda(torch::Tensor out) {
    const int64_t n = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    demon_attack_noop_kernel<<<blocks, threads>>>(out.data_ptr<unsigned char>(), n);
}

void demon_attack_decode_ram_cuda(
    torch::Tensor ram,
    torch::Tensor previous_score,
    torch::Tensor reward,
    torch::Tensor score,
    torch::Tensor lives,
    torch::Tensor terminal) {
    const int64_t batch_size = ram.size(0);
    const int threads = 256;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    demon_attack_decode_ram_kernel<<<blocks, threads>>>(
        ram.data_ptr<unsigned char>(),
        previous_score.data_ptr<int>(),
        reward.data_ptr<int>(),
        score.data_ptr<int>(),
        lives.data_ptr<int>(),
        terminal.data_ptr<bool>(),
        batch_size);
}

void demon_attack_reset_indices_cuda(
    torch::Tensor indices,
    torch::Tensor states,
    torch::Tensor frame_states,
    torch::Tensor ram,
    torch::Tensor cached_states,
    torch::Tensor cached_frame_states,
    torch::Tensor cached_ram,
    torch::Tensor rand_states,
    torch::Tensor cache_index,
    int64_t state_size,
    int64_t frame_state_size,
    int64_t ram_size,
    int64_t noop_reset_steps) {
    const int64_t count = indices.numel();
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    demon_attack_reset_indices_kernel<<<blocks, threads>>>(
        indices.data_ptr<int64_t>(),
        states.data_ptr<unsigned char>(),
        frame_states.data_ptr<unsigned char>(),
        ram.data_ptr<unsigned char>(),
        cached_states.data_ptr<unsigned char>(),
        cached_frame_states.data_ptr<unsigned char>(),
        cached_ram.data_ptr<unsigned char>(),
        reinterpret_cast<unsigned int*>(rand_states.data_ptr<int>()),
        reinterpret_cast<unsigned int*>(cache_index.data_ptr<int>()),
        state_size,
        frame_state_size,
        ram_size,
        noop_reset_steps,
        count);
}
