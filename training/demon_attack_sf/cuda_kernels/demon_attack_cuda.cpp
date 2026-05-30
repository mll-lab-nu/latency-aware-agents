#include <torch/extension.h>

void demon_attack_noop_cuda(torch::Tensor out);
void demon_attack_decode_ram_cuda(
    torch::Tensor ram,
    torch::Tensor previous_score,
    torch::Tensor reward,
    torch::Tensor score,
    torch::Tensor lives,
    torch::Tensor terminal);
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
    int64_t noop_reset_steps);

void demon_attack_noop(torch::Tensor out) {
    demon_attack_noop_cuda(out);
}

void demon_attack_decode_ram(
    torch::Tensor ram,
    torch::Tensor previous_score,
    torch::Tensor reward,
    torch::Tensor score,
    torch::Tensor lives,
    torch::Tensor terminal) {
    demon_attack_decode_ram_cuda(ram, previous_score, reward, score, lives, terminal);
}

void demon_attack_reset_indices(
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
    demon_attack_reset_indices_cuda(
        indices,
        states,
        frame_states,
        ram,
        cached_states,
        cached_frame_states,
        cached_ram,
        rand_states,
        cache_index,
        state_size,
        frame_state_size,
        ram_size,
        noop_reset_steps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("demon_attack_noop", &demon_attack_noop, "Demon Attack CUDA extension smoke kernel");
    m.def("demon_attack_decode_ram", &demon_attack_decode_ram, "Decode Demon Attack RAM reward and terminal state");
    m.def("demon_attack_reset_indices", &demon_attack_reset_indices, "Reset selected Demon Attack CuLE states");
}
