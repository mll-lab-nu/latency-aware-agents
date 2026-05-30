#include <cule/cule.hpp>
#include <cule/cuda.hpp>

#include <torchcule/atari_env.hpp>
#include <torchcule/atari_state.hpp>
#include <torchcule/atari_state.cpp>

#include <cmath>
#include <cstdint>

using cule_policy = cule::cuda::parallel_execution_policy;

namespace
{

constexpr double FRAME_EPS = 1e-10;
constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr uint64_t SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15ULL;

__device__ __forceinline__ uint64_t splitmix64_next(int64_t* __restrict__ rng_state, int env)
{
    uint64_t state = static_cast<uint64_t>(rng_state[env]) + SPLITMIX64_INCREMENT;
    rng_state[env] = static_cast<int64_t>(state);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ double rng_uniform01(int64_t* __restrict__ rng_state, int env)
{
    return static_cast<double>(splitmix64_next(rng_state, env) >> 11) * (1.0 / 9007199254740992.0);
}

enum DemonDecisionPackedColumn : int
{
    DEMON_DECISION_REWARD = 0,
    DEMON_DECISION_TERMINATED = 1,
    DEMON_DECISION_TRUNCATED = 2,
    DEMON_DECISION_SCORE = 3,
    DEMON_DECISION_RAW_FRAMES_ADVANCED = 4,
    DEMON_DECISION_LIVES = 5,
    DEMON_DECISION_ENV_STEP_IDX = 6,
    DEMON_DECISION_DECISION_STEP_IDX = 7,
    DEMON_DECISION_LATENCY_RAW_FRAMES = 8,
    DEMON_DECISION_APPLIED_ACTION = 9,
    DEMON_DECISION_PENDING_COUNT = 10,
    DEMON_DECISION_READY_RAW_FRAME = 11,
    DEMON_DECISION_QUEUED_ACTION = 12,
    DEMON_DECISION_TIMEOUT = 13,
    DEMON_DECISION_COLUMN_COUNT = 14,
};

__device__ __forceinline__ int64_t frames_float_to_raw_frames_cuda(double frames)
{
    return static_cast<int64_t>(ceil(frames - FRAME_EPS));
}

__device__ __forceinline__ int64_t sample_latency_raw_frames(
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    int64_t* __restrict__ latency_rng_state,
    int env)
{
    if (latency_distribution_type == 0)
    {
        return fixed_latency_raw_frames;
    }

    double sampled = latency_distribution_min;
    if (latency_distribution_type == 1)
    {
        sampled = latency_distribution_min + (
            latency_distribution_max - latency_distribution_min) * rng_uniform01(latency_rng_state, env);
    }
    else
    {
        if (latency_distribution_std == 0.0)
        {
            sampled = latency_distribution_mean;
        }
        else
        {
            double u1 = rng_uniform01(latency_rng_state, env);
            const double u2 = rng_uniform01(latency_rng_state, env);
            if (u1 < 1e-12)
            {
                u1 = 1e-12;
            }
            sampled = latency_distribution_mean + latency_distribution_std * sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
        }
    }
    if (sampled < latency_distribution_min)
    {
        sampled = latency_distribution_min;
    }
    if (sampled > latency_distribution_max)
    {
        sampled = latency_distribution_max;
    }
    return frames_float_to_raw_frames_cuda(sampled);
}

__global__ void demon_decision_begin_kernel(
    const int64_t* __restrict__ actions,
    const bool* __restrict__ active,
    int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ next_action_id,
    int64_t* __restrict__ current_latency_raw_frames,
    int64_t* __restrict__ current_applied_action,
    int64_t* __restrict__ persistence_remaining,
    int64_t* __restrict__ pending_actions,
    int64_t* __restrict__ pending_ready_raw_frames,
    int64_t* __restrict__ pending_action_ids,
    int64_t* __restrict__ pending_head,
    int64_t* __restrict__ pending_count,
    int64_t* __restrict__ decision_queued_actions,
    int64_t* __restrict__ decision_latency_raw_frames,
    int64_t* __restrict__ decision_ready_raw_frames,
    bool* __restrict__ decision_timeout,
    float* __restrict__ decision_rewards,
    bool* __restrict__ decision_terminated,
    bool* __restrict__ decision_truncated,
    int64_t* __restrict__ decision_raw_frames_advanced,
    bool* __restrict__ still_active,
    int64_t* __restrict__ latency_rng_state,
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    int batch_size,
    int queue_capacity)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size)
    {
        return;
    }

    decision_timeout[env] = false;
    decision_rewards[env] = 0.0f;
    decision_terminated[env] = false;
    decision_truncated[env] = false;
    decision_raw_frames_advanced[env] = 0;
    still_active[env] = active[env];

    if (!active[env])
    {
        return;
    }

    const int64_t latency = sample_latency_raw_frames(
        fixed_latency_raw_frames,
        latency_distribution_type,
        latency_distribution_mean,
        latency_distribution_std,
        latency_distribution_min,
        latency_distribution_max,
        latency_rng_state,
        env);
    const int64_t ready_raw_frame = env_step_idx[env] + latency;
    current_latency_raw_frames[env] = latency;
    decision_latency_raw_frames[env] = latency;
    decision_ready_raw_frames[env] = ready_raw_frame;
    decision_queued_actions[env] = actions[env];

    const int64_t tail = (pending_head[env] + pending_count[env]) % queue_capacity;
    const int64_t base = static_cast<int64_t>(env) * queue_capacity;
    pending_actions[base + tail] = actions[env];
    pending_ready_raw_frames[base + tail] = ready_raw_frame;
    pending_action_ids[base + tail] = next_action_id[env];
    pending_count[env] += 1;
    next_action_id[env] += 1;
}

__global__ void demon_decision_select_actions_kernel(
    const bool* __restrict__ still_active,
    const int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ current_applied_action,
    int64_t* __restrict__ persistence_remaining,
    int64_t* __restrict__ pending_actions,
    int64_t* __restrict__ pending_ready_raw_frames,
    int64_t* __restrict__ pending_action_ids,
    int64_t* __restrict__ pending_head,
    int64_t* __restrict__ pending_count,
    int64_t* __restrict__ scratch_actions,
    int64_t* __restrict__ scratch_ready_raw_frames,
    int64_t* __restrict__ scratch_action_ids,
    const cule::atari::Action* __restrict__ action_set,
    cule::atari::Action* __restrict__ raw_action_values,
    bool* __restrict__ doneBuffer,
    float* __restrict__ step_rewards,
    int64_t hold_duration,
    int64_t ordering_policy,
    int batch_size,
    int queue_capacity)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size)
    {
        return;
    }

    doneBuffer[env] = !still_active[env];
    step_rewards[env] = 0.0f;
    raw_action_values[env] = action_set[0];
    if (!still_active[env])
    {
        return;
    }

    const int64_t base = static_cast<int64_t>(env) * queue_capacity;
    bool selected = false;
    int64_t selected_action = 0;
    int64_t count = pending_count[env];
    const int64_t head = pending_head[env];
    const int64_t current_step = env_step_idx[env];

    if (count > 0)
    {
        if (ordering_policy == 0)
        {
            const int64_t pos = head;
            if (pending_ready_raw_frames[base + pos] <= current_step)
            {
                selected = true;
                selected_action = pending_actions[base + pos];
                pending_head[env] = (head + 1) % queue_capacity;
                pending_count[env] = count - 1;
            }
        }
        else
        {
            int64_t selected_offset = -1;
            int64_t keep_count = 0;
            for (int64_t offset = 0; offset < count; ++offset)
            {
                const int64_t pos = (head + offset) % queue_capacity;
                const bool ready = pending_ready_raw_frames[base + pos] <= current_step;
                if (ready)
                {
                    selected_offset = offset;
                }
                else
                {
                    scratch_actions[base + keep_count] = pending_actions[base + pos];
                    scratch_ready_raw_frames[base + keep_count] = pending_ready_raw_frames[base + pos];
                    scratch_action_ids[base + keep_count] = pending_action_ids[base + pos];
                    keep_count += 1;
                }
            }
            if (selected_offset >= 0)
            {
                const int64_t selected_pos = (head + selected_offset) % queue_capacity;
                selected = true;
                selected_action = pending_actions[base + selected_pos];
                for (int64_t offset = 0; offset < keep_count; ++offset)
                {
                    pending_actions[base + offset] = scratch_actions[base + offset];
                    pending_ready_raw_frames[base + offset] = scratch_ready_raw_frames[base + offset];
                    pending_action_ids[base + offset] = scratch_action_ids[base + offset];
                }
                pending_head[env] = 0;
                pending_count[env] = keep_count;
            }
        }
    }

    if (selected)
    {
        current_applied_action[env] = selected_action;
        persistence_remaining[env] = hold_duration;
    }
    int64_t raw_action = 0;
    if (persistence_remaining[env] > 0)
    {
        raw_action = current_applied_action[env];
        persistence_remaining[env] -= 1;
    }
    current_applied_action[env] = raw_action;
    raw_action_values[env] = action_set[raw_action];
}

__global__ void demon_decision_accumulate_kernel(
    const bool* __restrict__ active,
    bool* __restrict__ still_active,
    const bool* __restrict__ doneBuffer,
    const float* __restrict__ step_rewards,
    int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ episode_step_idx,
    int32_t* __restrict__ scoresBuffer,
    bool* __restrict__ decision_timeout,
    float* __restrict__ decision_rewards,
    bool* __restrict__ decision_terminated,
    bool* __restrict__ decision_truncated,
    int64_t* __restrict__ decision_raw_frames_advanced,
    int64_t max_episode_steps,
    int batch_size)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || !still_active[env])
    {
        return;
    }

    const float reward = step_rewards[env];
    decision_rewards[env] += reward;
    scoresBuffer[env] += static_cast<int32_t>(reward);
    decision_raw_frames_advanced[env] += 1;
    env_step_idx[env] += 1;
    episode_step_idx[env] += 1;

    const bool terminated = doneBuffer[env] && active[env];
    bool truncated = false;
    if (max_episode_steps > 0 && episode_step_idx[env] >= max_episode_steps)
    {
        truncated = true;
        decision_timeout[env] = true;
    }
    decision_terminated[env] = decision_terminated[env] || terminated;
    decision_truncated[env] = decision_truncated[env] || truncated;
    still_active[env] = !(decision_terminated[env] || decision_truncated[env]);
}

__global__ void demon_decision_finish_kernel(
    const bool* __restrict__ active,
    int64_t* __restrict__ decision_step_idx,
    const int64_t* __restrict__ env_step_idx,
    const int64_t* __restrict__ current_applied_action,
    const int64_t* __restrict__ pending_count,
    const int64_t* __restrict__ decision_queued_actions,
    const int64_t* __restrict__ decision_latency_raw_frames,
    const int64_t* __restrict__ decision_ready_raw_frames,
    const bool* __restrict__ decision_timeout,
    const float* __restrict__ decision_rewards,
    const bool* __restrict__ decision_terminated,
    const bool* __restrict__ decision_truncated,
    const int64_t* __restrict__ decision_raw_frames_advanced,
    const int32_t* __restrict__ scoresBuffer,
    const int32_t* __restrict__ livesBuffer,
    double* __restrict__ packed_decision,
    int batch_size)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size)
    {
        return;
    }

    if (active[env])
    {
        decision_step_idx[env] += 1;
    }

    const int packed_idx = env * DEMON_DECISION_COLUMN_COUNT;
    packed_decision[packed_idx + DEMON_DECISION_REWARD] = static_cast<double>(decision_rewards[env]);
    packed_decision[packed_idx + DEMON_DECISION_TERMINATED] = decision_terminated[env] ? 1.0 : 0.0;
    packed_decision[packed_idx + DEMON_DECISION_TRUNCATED] = decision_truncated[env] ? 1.0 : 0.0;
    packed_decision[packed_idx + DEMON_DECISION_SCORE] = static_cast<double>(scoresBuffer[env]);
    packed_decision[packed_idx + DEMON_DECISION_RAW_FRAMES_ADVANCED] =
        static_cast<double>(decision_raw_frames_advanced[env]);
    packed_decision[packed_idx + DEMON_DECISION_LIVES] = static_cast<double>(livesBuffer[env]);
    packed_decision[packed_idx + DEMON_DECISION_ENV_STEP_IDX] = static_cast<double>(env_step_idx[env]);
    packed_decision[packed_idx + DEMON_DECISION_DECISION_STEP_IDX] = static_cast<double>(decision_step_idx[env]);
    packed_decision[packed_idx + DEMON_DECISION_LATENCY_RAW_FRAMES] =
        static_cast<double>(decision_latency_raw_frames[env]);
    packed_decision[packed_idx + DEMON_DECISION_APPLIED_ACTION] = static_cast<double>(current_applied_action[env]);
    packed_decision[packed_idx + DEMON_DECISION_PENDING_COUNT] = static_cast<double>(pending_count[env]);
    packed_decision[packed_idx + DEMON_DECISION_READY_RAW_FRAME] =
        static_cast<double>(decision_ready_raw_frames[env]);
    packed_decision[packed_idx + DEMON_DECISION_QUEUED_ACTION] = static_cast<double>(decision_queued_actions[env]);
    packed_decision[packed_idx + DEMON_DECISION_TIMEOUT] = decision_timeout[env] ? 1.0 : 0.0;
}

} // namespace

AtariEnv::
AtariEnv(const cule::atari::rom& cart,
         const size_t num_envs,
         const size_t noop_reset_steps)
    : super_t(cart, num_envs, noop_reset_steps),
      use_cuda(false),
      gpu_id(-1),
      cule_par(new agency::parallel_execution_policy())
{
}

AtariEnv::
~AtariEnv()
{
    if(use_cuda)
    {
        delete &get_policy<cule_policy>();
    }
    else
    {
        delete &get_policy<agency::parallel_execution_policy>();
    }
}

void
AtariEnv::
reset(uint32_t* seedBuffer)
{
    if(use_cuda)
    {
        super_t::reset(get_policy<cule_policy>(), seedBuffer);
    }
    else
    {
        super_t::reset(get_policy<agency::parallel_execution_policy>(), seedBuffer);
    }
}

void
AtariEnv::
reset_states()
{
    if(use_cuda)
    {
        super_t::reset_states(get_policy<cule_policy>());
    }
    else
    {
        super_t::reset_states(get_policy<agency::parallel_execution_policy>());
    }
}

void
AtariEnv::
get_states(const size_t num_states,
           const int32_t* indices_ptr,
           AtariState* states_ptr)
{
    agency::vector<cule::atari::state> output_states(num_states);
    agency::vector<cule::atari::frame_state> output_frame_states(num_states);
    agency::vector<uint8_t> output_states_ram(256 * num_states);

    if(use_cuda)
    {
        agency::vector<cule::atari::state, agency::cuda::allocator<cule::atari::state>> output_states_gpu(num_states);
        agency::vector<cule::atari::frame_state, agency::cuda::allocator<cule::atari::frame_state>> output_frame_states_gpu(num_states);
        agency::vector<uint8_t, agency::cuda::allocator<uint8_t>> output_states_ram_gpu(256 * num_states);
        agency::vector<int32_t, agency::cuda::allocator<int32_t>> indices_gpu(indices_ptr, indices_ptr + num_states);

        super_t::get_states(get_policy<cule_policy>(), num_states, indices_gpu.data(), output_states_gpu.data(), output_frame_states_gpu.data(), output_states_ram_gpu.data());
        get_policy<cule_policy>().sync();

        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_states_gpu.begin(), output_states_gpu.end(), output_states.begin());
        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_frame_states_gpu.begin(), output_frame_states_gpu.end(), output_frame_states.begin());
        agency::detail::copy(get_policy<agency::parallel_execution_policy>(), output_states_ram_gpu.begin(), output_states_ram_gpu.end(), output_states_ram.begin());
        get_policy<cule_policy>().sync();
    }
    else
    {
        super_t::get_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr, output_states.data(), output_frame_states.data(), nullptr);
    }

    agency::bulk_invoke(get_policy<agency::parallel_execution_policy>()(num_states),
                        decode_states_functor{},
                        use_cuda,
                        this->cart,
                        states_ptr,
                        output_states.data(),
                        output_frame_states.data(),
                        output_states_ram.data());
}

void
AtariEnv::
set_states(const size_t num_states,
           const int32_t* indices_ptr,
           const AtariState* states_ptr)
{
    agency::vector<cule::atari::state> input_states(num_states);
    agency::vector<cule::atari::frame_state> input_frame_states(num_states);
    agency::vector<uint8_t> input_states_ram(256 * num_states);

    agency::bulk_invoke(agency::seq(num_states),
                        encode_states_functor{},
                        this->cart,
                        states_ptr,
                        input_states.data(),
                        input_frame_states.data(),
                        input_states_ram.data());

    if(use_cuda)
    {
        agency::vector<cule::atari::state, agency::cuda::allocator<cule::atari::state>> input_states_gpu(input_states.begin(), input_states.end());
        agency::vector<cule::atari::frame_state, agency::cuda::allocator<cule::atari::frame_state>> input_frame_states_gpu(input_frame_states.begin(), input_frame_states.end());
        agency::vector<uint8_t, agency::cuda::allocator<uint8_t>> input_states_ram_gpu(input_states_ram.begin(), input_states_ram.end());
        agency::vector<int32_t, agency::cuda::allocator<int32_t>> indices_gpu(indices_ptr, indices_ptr + num_states);

        super_t::set_states(get_policy<cule_policy>(), num_states, indices_gpu.data(), input_states_gpu.data(),
                            input_frame_states_gpu.data(), input_states_ram_gpu.data());
        get_policy<cule_policy>().sync();
    }
    else
    {
        super_t::set_states(get_policy<agency::parallel_execution_policy>(), num_states, indices_ptr,
                            input_states.data(), input_frame_states.data(), input_states_ram.data());
    }
}

void
AtariEnv::
step(const bool fire_reset,
     const cule::atari::Action* playerABuffer,
     const cule::atari::Action* playerBBuffer,
     bool* doneBuffer)
{
    if(use_cuda)
    {
        super_t::step(get_policy<cule_policy>(), fire_reset, playerABuffer, playerBBuffer, doneBuffer);
    }
    else
    {
        super_t::step(get_policy<agency::parallel_execution_policy>(), fire_reset, playerABuffer, playerBBuffer, doneBuffer);
    }
}

void
AtariEnv::
get_data(const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    if(use_cuda)
    {
        super_t::get_data(get_policy<cule_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
    else
    {
        super_t::get_data(get_policy<agency::parallel_execution_policy>(), episodic_life, doneBuffer, rewardsBuffer, livesBuffer);
    }
}

void
AtariEnv::
generate_frames(const bool rescale,
                const bool last_frame,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    if(use_cuda)
    {
        super_t::generate_frames(get_policy<cule_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
    else
    {
        super_t::generate_frames(get_policy<agency::parallel_execution_policy>(), rescale, last_frame, num_channels, imageBuffer);
    }
}

void
AtariEnv::
step_decision(const bool episodic_life,
              const size_t obs_stride_raw_frames,
              const size_t num_channels,
              const int64_t hold_duration,
              const int64_t ordering_policy,
              const int64_t max_episode_steps,
              const int64_t fixed_latency_raw_frames,
              const int64_t latency_distribution_type,
              const double latency_distribution_mean,
              const double latency_distribution_std,
              const double latency_distribution_min,
              const double latency_distribution_max,
              const int64_t batch_size,
              const int64_t queue_capacity,
              const int64_t* actions,
              const bool* active,
              int64_t* env_step_idx,
              int64_t* decision_step_idx,
              int64_t* next_action_id,
              int64_t* current_latency_raw_frames,
              int64_t* current_applied_action,
              int64_t* persistence_remaining,
              int64_t* pending_actions,
              int64_t* pending_ready_raw_frames,
              int64_t* pending_action_ids,
              int64_t* pending_head,
              int64_t* pending_count,
              int64_t* scratch_actions,
              int64_t* scratch_ready_raw_frames,
              int64_t* scratch_action_ids,
              int64_t* decision_queued_actions,
              int64_t* decision_latency_raw_frames,
              int64_t* decision_ready_raw_frames,
              bool* decision_timeout,
              double* packed_decision,
              float* decision_rewards,
              bool* decision_terminated,
              bool* decision_truncated,
              int64_t* decision_raw_frames_advanced,
              int64_t* episode_step_idx,
              int64_t* latency_rng_state,
              cule::atari::Action* action_set,
              cule::atari::Action* raw_action_values,
              bool* still_active,
              bool* doneBuffer,
              float* step_rewards,
              int32_t* livesBuffer,
              int32_t* scoresBuffer,
              uint8_t* imageBuffer)
{
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    cudaStream_t stream = get_policy<cule_policy>().getStream();

    demon_decision_begin_kernel<<<blocks, threads, 0, stream>>>(
        actions,
        active,
        env_step_idx,
        next_action_id,
        current_latency_raw_frames,
        current_applied_action,
        persistence_remaining,
        pending_actions,
        pending_ready_raw_frames,
        pending_action_ids,
        pending_head,
        pending_count,
        decision_queued_actions,
        decision_latency_raw_frames,
        decision_ready_raw_frames,
        decision_timeout,
        decision_rewards,
        decision_terminated,
        decision_truncated,
        decision_raw_frames_advanced,
        still_active,
        latency_rng_state,
        fixed_latency_raw_frames,
        latency_distribution_type,
        latency_distribution_mean,
        latency_distribution_std,
        latency_distribution_min,
        latency_distribution_max,
        static_cast<int>(batch_size),
        static_cast<int>(queue_capacity));

    for (size_t raw_step = 0; raw_step < obs_stride_raw_frames; ++raw_step)
    {
        demon_decision_select_actions_kernel<<<blocks, threads, 0, stream>>>(
            still_active,
            env_step_idx,
            current_applied_action,
            persistence_remaining,
            pending_actions,
            pending_ready_raw_frames,
            pending_action_ids,
            pending_head,
            pending_count,
            scratch_actions,
            scratch_ready_raw_frames,
            scratch_action_ids,
            action_set,
            raw_action_values,
            doneBuffer,
            step_rewards,
            hold_duration,
            ordering_policy,
            static_cast<int>(batch_size),
            static_cast<int>(queue_capacity));
        super_t::step(get_policy<cule_policy>(), false, raw_action_values, nullptr, doneBuffer);
        super_t::get_data(get_policy<cule_policy>(), episodic_life, doneBuffer, step_rewards, livesBuffer);
        demon_decision_accumulate_kernel<<<blocks, threads, 0, stream>>>(
            active,
            still_active,
            doneBuffer,
            step_rewards,
            env_step_idx,
            episode_step_idx,
            scoresBuffer,
            decision_timeout,
            decision_rewards,
            decision_terminated,
            decision_truncated,
            decision_raw_frames_advanced,
            max_episode_steps,
            static_cast<int>(batch_size));
        super_t::reset_states(get_policy<cule_policy>());
    }

    super_t::generate_frames(get_policy<cule_policy>(), true, true, num_channels, imageBuffer);
    demon_decision_finish_kernel<<<blocks, threads, 0, stream>>>(
        active,
        decision_step_idx,
        env_step_idx,
        current_applied_action,
        pending_count,
        decision_queued_actions,
        decision_latency_raw_frames,
        decision_ready_raw_frames,
        decision_timeout,
        decision_rewards,
        decision_terminated,
        decision_truncated,
        decision_raw_frames_advanced,
        scoresBuffer,
        livesBuffer,
        packed_decision,
        static_cast<int>(batch_size));
}

void
AtariEnv::
generate_random_actions(cule::atari::Action* actionBuffer)
{
    if(use_cuda)
    {
        super_t::generate_random_actions(get_policy<cule_policy>(), actionBuffer);
    }
    else
    {
        super_t::generate_random_actions(get_policy<agency::parallel_execution_policy>(), actionBuffer);
    }
}

void
AtariEnv::
sync_other_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_other_stream(stream);
    }
}

void
AtariEnv::
sync_this_stream(cudaStream_t& stream)
{
    if(use_cuda)
    {
        get_policy<cule_policy>().insert_this_stream(stream);
    }
}

void
AtariEnv::
set_cuda(const bool use_cuda, const int32_t gpu_id)
{
    if(this->use_cuda != use_cuda)
    {
        this->use_cuda = use_cuda;

        if(use_cuda)
        {
            this->gpu_id = gpu_id;
            CULE_ERRCHK(cudaSetDevice(gpu_id));
            delete &get_policy<agency::parallel_execution_policy>();
            cule_par = new cule_policy();
        }
        else
        {
            this->gpu_id = -1;
            delete &get_policy<cule_policy>();
            cule_par = new agency::parallel_execution_policy();
        }
    }
}

template<typename ExecutionPolicy>
ExecutionPolicy&
AtariEnv::
get_policy()
{
    assert(cule_par != nullptr);
    if(gpu_id != -1)
    {
        CULE_ERRCHK(cudaSetDevice(gpu_id));
    }

    return *reinterpret_cast<ExecutionPolicy*>(cule_par);
}

size_t
AtariEnv::
state_size()
{
    return sizeof(cule::atari::state);
}

size_t
AtariEnv::
frame_state_size()
{
    return sizeof(cule::atari::frame_state);
}

size_t
AtariEnv::
tia_update_size()
{
    return cule::atari::ENV_UPDATE_SIZE;
}

#include <cule/atari/rom.cpp>
#include <cule/atari/wrapper.cpp>
