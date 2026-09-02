#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <torch/extension.h>

namespace {

constexpr int PIPE_VEL_X = -4;
constexpr int PLAYER_MAX_VEL_Y = 10;
constexpr int PLAYER_ACC_Y = 1;
constexpr int PLAYER_FLAP_ACC = -9;
constexpr int PLAYER_HEIGHT = 24;
constexpr int PLAYER_WIDTH = 34;
constexpr int PIPE_WIDTH = 52;
constexpr int PIPE_HEIGHT = 320;
constexpr int PLAYER_ROT_THR = 20;
constexpr int PLAYER_VEL_ROT = 3;
constexpr double FRAME_EPS = 1e-10;
constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr uint64_t SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15ULL;

__device__ __forceinline__ bool rects_collide(
    int64_t player_x,
    int64_t player_y,
    int64_t pipe_x,
    int64_t pipe_y) {
    return (
        (player_x < pipe_x + PIPE_WIDTH) &&
        (player_x + PLAYER_WIDTH > pipe_x) &&
        (player_y < pipe_y + PIPE_HEIGHT) &&
        (player_y + PLAYER_HEIGHT > pipe_y));
}

__device__ __forceinline__ int trunc_to_int(double value) {
    return static_cast<int>(value);
}

__device__ __forceinline__ uint64_t splitmix64_next(int64_t* __restrict__ rng_state, int env) {
    uint64_t state = static_cast<uint64_t>(rng_state[env]) + SPLITMIX64_INCREMENT;
    rng_state[env] = static_cast<int64_t>(state);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ double rng_uniform01(int64_t* __restrict__ rng_state, int env) {
    return static_cast<double>(splitmix64_next(rng_state, env) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ int64_t frames_float_to_raw_frames_cuda(double frames) {
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
    int env) {
    if (latency_distribution_type == 0) {
        return fixed_latency_raw_frames;
    }

    double sampled = latency_distribution_min;
    if (latency_distribution_type == 1) {
        sampled = latency_distribution_min + (
            latency_distribution_max - latency_distribution_min) * rng_uniform01(latency_rng_state, env);
    } else {
        if (latency_distribution_std == 0.0) {
            sampled = latency_distribution_mean;
        } else {
            double u1 = rng_uniform01(latency_rng_state, env);
            const double u2 = rng_uniform01(latency_rng_state, env);
            if (u1 < 1e-12) {
                u1 = 1e-12;
            }
            sampled = latency_distribution_mean + latency_distribution_std * sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
        }
    }
    if (sampled < latency_distribution_min) {
        sampled = latency_distribution_min;
    }
    if (sampled > latency_distribution_max) {
        sampled = latency_distribution_max;
    }
    return frames_float_to_raw_frames_cuda(sampled);
}

__device__ __forceinline__ void maybe_blit(
    const unsigned char* __restrict__ rgb,
    const bool* __restrict__ mask,
    int sprite_h,
    int sprite_w,
    int sprite_stride_w,
    int sprite_x,
    int sprite_y,
    int x,
    int y,
    unsigned char& r,
    unsigned char& g,
    unsigned char& b) {
    const int sx = x - sprite_x;
    const int sy = y - sprite_y;
    if (sx < 0 || sy < 0 || sx >= sprite_w || sy >= sprite_h) {
        return;
    }
    const int pix = sy * sprite_stride_w + sx;
    if (!mask[pix]) {
        return;
    }
    const int rgb_idx = pix * 3;
    r = rgb[rgb_idx + 0];
    g = rgb[rgb_idx + 1];
    b = rgb[rgb_idx + 2];
}

__global__ void flappy_step_kernel(
    double* __restrict__ player_x,
    double* __restrict__ player_y,
    double* __restrict__ player_vel_y,
    int64_t* __restrict__ player_rot,
    int64_t* __restrict__ player_idx,
    int64_t* __restrict__ player_idx_cycle_pos,
    int64_t* __restrict__ loop_iter,
    int64_t* __restrict__ score,
    const int64_t* __restrict__ score_limit,
    double* __restrict__ ground_x,
    const double* __restrict__ ground_y,
    double* __restrict__ upper_pipe_xy,
    double* __restrict__ lower_pipe_xy,
    const int64_t* __restrict__ actions,
    const bool* __restrict__ active,
    double* __restrict__ rewards,
    bool* __restrict__ terminated,
    bool* __restrict__ truncated,
    bool* __restrict__ offscreen,
    double* __restrict__ packed_step,
    int batch_size,
    int base_shift) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }

    rewards[env] = 0.0;
    terminated[env] = false;
    truncated[env] = false;
    offscreen[env * 3 + 0] = false;
    offscreen[env * 3 + 1] = false;
    offscreen[env * 3 + 2] = false;

    if (!active[env]) {
        const int packed_idx = env * 7;
        packed_step[packed_idx + 0] = 0.0;
        packed_step[packed_idx + 1] = 0.0;
        packed_step[packed_idx + 2] = 0.0;
        packed_step[packed_idx + 3] = static_cast<double>(score[env]);
        packed_step[packed_idx + 4] = 0.0;
        packed_step[packed_idx + 5] = 0.0;
        packed_step[packed_idx + 6] = 0.0;
        return;
    }

    rewards[env] = 0.1;
    const bool flap = (
        actions[env] == 1 &&
        player_y[env] > -2.0 * static_cast<double>(PLAYER_HEIGHT));
    if (flap) {
        player_vel_y[env] = static_cast<double>(PLAYER_FLAP_ACC);
    }

    const double player_mid_pos = player_x[env] + static_cast<double>(PLAYER_WIDTH) / 2.0;
    bool scored = false;
    for (int pipe = 0; pipe < 3; ++pipe) {
        const int xy_idx = (env * 3 + pipe) * 2;
        const double pipe_mid_pos = upper_pipe_xy[xy_idx + 0] + static_cast<double>(PIPE_WIDTH) / 2.0;
        const bool passed = (
            pipe_mid_pos <= player_mid_pos &&
            player_mid_pos < pipe_mid_pos + 4.0);
        if (passed) {
            score[env] += 1;
            scored = true;
        }
    }
    if (scored) {
        rewards[env] = 1.0;
    }

    if (((loop_iter[env] + 1) % 3) == 0) {
        const int64_t cycle_pos = player_idx_cycle_pos[env];
        int64_t next_idx = 0;
        if (cycle_pos == 0) {
            next_idx = 0;
        } else if (cycle_pos == 1) {
            next_idx = 1;
        } else if (cycle_pos == 2) {
            next_idx = 2;
        } else {
            next_idx = 1;
        }
        player_idx[env] = next_idx;
        player_idx_cycle_pos[env] = (cycle_pos + 1) % 4;
    }

    loop_iter[env] = (loop_iter[env] + 1) % 30;
    ground_x[env] = -fmod((-ground_x[env] + 100.0), static_cast<double>(base_shift));

    if (player_rot[env] > -90) {
        player_rot[env] -= PLAYER_VEL_ROT;
    }

    if (!flap && player_vel_y[env] < static_cast<double>(PLAYER_MAX_VEL_Y)) {
        player_vel_y[env] += static_cast<double>(PLAYER_ACC_Y);
    }
    if (flap) {
        player_rot[env] = 45;
    }

    const double max_y_delta = ground_y[env] - player_y[env] - static_cast<double>(PLAYER_HEIGHT);
    const double delta = player_vel_y[env] < max_y_delta ? player_vel_y[env] : max_y_delta;
    player_y[env] += delta;

    for (int pipe = 0; pipe < 3; ++pipe) {
        const int xy_idx = (env * 3 + pipe) * 2;
        upper_pipe_xy[xy_idx + 0] += static_cast<double>(PIPE_VEL_X);
        lower_pipe_xy[xy_idx + 0] += static_cast<double>(PIPE_VEL_X);
        if (upper_pipe_xy[xy_idx + 0] < -static_cast<double>(PIPE_WIDTH)) {
            offscreen[env * 3 + pipe] = true;
        }
    }

    if (player_y[env] < 0.0) {
        rewards[env] = -0.5;
    }

    const bool ground = player_y[env] + static_cast<double>(PLAYER_HEIGHT) >= ground_y[env] - 1.0;
    bool crashed = ground;
    if (!ground) {
        const int64_t px = static_cast<int64_t>(player_x[env]);
        const int64_t py = static_cast<int64_t>(player_y[env]);
        for (int pipe = 0; pipe < 3; ++pipe) {
            const int xy_idx = (env * 3 + pipe) * 2;
            const int64_t up_x = static_cast<int64_t>(upper_pipe_xy[xy_idx + 0]);
            const int64_t up_y = static_cast<int64_t>(upper_pipe_xy[xy_idx + 1]);
            const int64_t low_x = static_cast<int64_t>(lower_pipe_xy[xy_idx + 0]);
            const int64_t low_y = static_cast<int64_t>(lower_pipe_xy[xy_idx + 1]);
            crashed = crashed || rects_collide(px, py, up_x, up_y) || rects_collide(px, py, low_x, low_y);
        }
    }
    if (crashed) {
        rewards[env] = -1.0;
        terminated[env] = true;
        player_vel_y[env] = 0.0;
    }

    if (score_limit[env] >= 0 && score[env] >= score_limit[env]) {
        truncated[env] = true;
    }

    const int packed_idx = env * 7;
    packed_step[packed_idx + 0] = rewards[env];
    packed_step[packed_idx + 1] = terminated[env] ? 1.0 : 0.0;
    packed_step[packed_idx + 2] = truncated[env] ? 1.0 : 0.0;
    packed_step[packed_idx + 3] = static_cast<double>(score[env]);
    packed_step[packed_idx + 4] = offscreen[env * 3 + 0] ? 1.0 : 0.0;
    packed_step[packed_idx + 5] = offscreen[env * 3 + 1] ? 1.0 : 0.0;
    packed_step[packed_idx + 6] = offscreen[env * 3 + 2] ? 1.0 : 0.0;
}

__global__ void flappy_render_kernel(
    unsigned char* __restrict__ out,
    const unsigned char* __restrict__ background_rgb,
    const unsigned char* __restrict__ fill_color,
    bool has_background,
    const unsigned char* __restrict__ base_rgb,
    const bool* __restrict__ base_mask,
    int base_h,
    int base_w,
    const unsigned char* __restrict__ pipe_upper_rgb,
    const bool* __restrict__ pipe_upper_mask,
    int pipe_upper_h,
    int pipe_upper_w,
    const unsigned char* __restrict__ pipe_lower_rgb,
    const bool* __restrict__ pipe_lower_mask,
    int pipe_lower_h,
    int pipe_lower_w,
    const unsigned char* __restrict__ bird_rgb,
    const bool* __restrict__ bird_mask,
    const int64_t* __restrict__ bird_heights,
    const int64_t* __restrict__ bird_widths,
    int bird_max_h,
    int bird_max_w,
    const double* __restrict__ player_x,
    const double* __restrict__ player_y,
    const int64_t* __restrict__ player_idx,
    const int64_t* __restrict__ player_rot,
    const double* __restrict__ ground_x,
    const double* __restrict__ ground_y,
    const double* __restrict__ upper_pipe_xy,
    const double* __restrict__ lower_pipe_xy,
    int batch_size,
    int screen_width,
    int screen_height,
    int num_rotations,
    int min_rotation,
    int64_t total_pixels) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total_pixels) {
        return;
    }

    const int x = static_cast<int>(linear % screen_width);
    const int y = static_cast<int>((linear / screen_width) % screen_height);
    const int env = static_cast<int>(linear / (screen_width * screen_height));

    unsigned char r;
    unsigned char g;
    unsigned char b;
    if (has_background) {
        const int bg_idx = (y * screen_width + x) * 3;
        r = background_rgb[bg_idx + 0];
        g = background_rgb[bg_idx + 1];
        b = background_rgb[bg_idx + 2];
    } else {
        r = fill_color[0];
        g = fill_color[1];
        b = fill_color[2];
    }

    for (int pipe = 0; pipe < 3; ++pipe) {
        const int xy_idx = (env * 3 + pipe) * 2;
        maybe_blit(
            pipe_upper_rgb,
            pipe_upper_mask,
            pipe_upper_h,
            pipe_upper_w,
            pipe_upper_w,
            trunc_to_int(upper_pipe_xy[xy_idx + 0]),
            trunc_to_int(upper_pipe_xy[xy_idx + 1]),
            x,
            y,
            r,
            g,
            b);
        maybe_blit(
            pipe_lower_rgb,
            pipe_lower_mask,
            pipe_lower_h,
            pipe_lower_w,
            pipe_lower_w,
            trunc_to_int(lower_pipe_xy[xy_idx + 0]),
            trunc_to_int(lower_pipe_xy[xy_idx + 1]),
            x,
            y,
            r,
            g,
            b);
    }

    maybe_blit(
        base_rgb,
        base_mask,
        base_h,
        base_w,
        base_w,
        trunc_to_int(ground_x[env]),
        trunc_to_int(ground_y[env]),
        x,
        y,
        r,
        g,
        b);

    int64_t rotation = player_rot[env];
    if (rotation > PLAYER_ROT_THR) {
        rotation = PLAYER_ROT_THR;
    }
    const int rotation_idx = static_cast<int>(rotation - min_rotation);
    const int bird_index = static_cast<int>(player_idx[env]) * num_rotations + rotation_idx;
    const int bird_h = static_cast<int>(bird_heights[bird_index]);
    const int bird_w = static_cast<int>(bird_widths[bird_index]);
    const unsigned char* bird_rgb_ptr = bird_rgb + static_cast<int64_t>(bird_index) * bird_max_h * bird_max_w * 3;
    const bool* bird_mask_ptr = bird_mask + static_cast<int64_t>(bird_index) * bird_max_h * bird_max_w;
    maybe_blit(
        bird_rgb_ptr,
        bird_mask_ptr,
        bird_h,
        bird_w,
        bird_max_w,
        trunc_to_int(player_x[env]),
        trunc_to_int(player_y[env]),
        x,
        y,
        r,
        g,
        b);

    const int64_t out_idx = linear * 3;
    out[out_idx + 0] = r;
    out[out_idx + 1] = g;
    out[out_idx + 2] = b;
}

__global__ void flappy_latency_begin_kernel(
    const int64_t* __restrict__ actions,
    const bool* __restrict__ active,
    const int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ next_action_id,
    int64_t* __restrict__ current_latency_raw_frames,
    int64_t* __restrict__ pending_actions,
    int64_t* __restrict__ pending_ready_raw_frames,
    int64_t* __restrict__ pending_action_ids,
    const int64_t* __restrict__ pending_head,
    int64_t* __restrict__ pending_count,
    int64_t* __restrict__ decision_queued_actions,
    int64_t* __restrict__ decision_latency_raw_frames,
    int64_t* __restrict__ decision_ready_raw_frames,
    int64_t* __restrict__ latency_rng_state,
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    const int64_t* __restrict__ profile_latency_raw_frames,
    int64_t use_profile_latency,
    int batch_size,
    int queue_capacity) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || !active[env]) {
        return;
    }

    const int64_t latency = use_profile_latency != 0
        ? profile_latency_raw_frames[env]
        : latency_distribution_type == 3
        ? current_latency_raw_frames[env]
        : sample_latency_raw_frames(
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

__global__ void flappy_latency_raw_actions_kernel(
    const bool* __restrict__ active,
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
    int64_t* __restrict__ raw_actions,
    int64_t hold_duration,
    int64_t ordering_policy,
    int batch_size,
    int queue_capacity) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }

    raw_actions[env] = 0;
    if (!active[env]) {
        return;
    }

    const int64_t base = static_cast<int64_t>(env) * queue_capacity;
    bool selected = false;
    int64_t selected_action = 0;
    int64_t count = pending_count[env];
    const int64_t head = pending_head[env];
    const int64_t current_step = env_step_idx[env];

    if (count > 0) {
        if (ordering_policy == 0) {
            const int64_t pos = head;
            if (pending_ready_raw_frames[base + pos] <= current_step) {
                selected = true;
                selected_action = pending_actions[base + pos];
                pending_head[env] = (head + 1) % queue_capacity;
                pending_count[env] = count - 1;
            }
        } else {
            int64_t selected_offset = -1;
            int64_t keep_count = 0;
            for (int64_t offset = 0; offset < count; ++offset) {
                const int64_t pos = (head + offset) % queue_capacity;
                const bool ready = pending_ready_raw_frames[base + pos] <= current_step;
                if (ready) {
                    selected_offset = offset;
                } else {
                    scratch_actions[base + keep_count] = pending_actions[base + pos];
                    scratch_ready_raw_frames[base + keep_count] = pending_ready_raw_frames[base + pos];
                    scratch_action_ids[base + keep_count] = pending_action_ids[base + pos];
                    keep_count += 1;
                }
            }
            if (selected_offset >= 0) {
                const int64_t selected_pos = (head + selected_offset) % queue_capacity;
                selected = true;
                selected_action = pending_actions[base + selected_pos];
                for (int64_t offset = 0; offset < keep_count; ++offset) {
                    pending_actions[base + offset] = scratch_actions[base + offset];
                    pending_ready_raw_frames[base + offset] = scratch_ready_raw_frames[base + offset];
                    pending_action_ids[base + offset] = scratch_action_ids[base + offset];
                }
                pending_head[env] = 0;
                pending_count[env] = keep_count;
            }
        }
    }

    if (selected) {
        current_applied_action[env] = selected_action;
        persistence_remaining[env] = hold_duration;
    }
    if (persistence_remaining[env] > 0) {
        raw_actions[env] = current_applied_action[env];
        persistence_remaining[env] -= 1;
    }
    current_applied_action[env] = raw_actions[env];
}

__global__ void flappy_latency_record_kernel(
    const bool* __restrict__ advanced,
    int64_t* __restrict__ env_step_idx,
    int batch_size) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < batch_size && advanced[env]) {
        env_step_idx[env] += 1;
    }
}

__global__ void flappy_latency_finish_kernel(
    const bool* __restrict__ active,
    int64_t* __restrict__ decision_step_idx,
    int batch_size) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < batch_size && active[env]) {
        decision_step_idx[env] += 1;
    }
}

__global__ void flappy_latency_core_decision_kernel(
    double* __restrict__ player_x,
    double* __restrict__ player_y,
    double* __restrict__ player_vel_y,
    int64_t* __restrict__ player_rot,
    int64_t* __restrict__ player_idx,
    int64_t* __restrict__ player_idx_cycle_pos,
    int64_t* __restrict__ loop_iter,
    int64_t* __restrict__ score,
    const int64_t* __restrict__ score_limit,
    double* __restrict__ ground_x,
    const double* __restrict__ ground_y,
    double* __restrict__ upper_pipe_xy,
    double* __restrict__ lower_pipe_xy,
    const int64_t* __restrict__ actions,
    const bool* __restrict__ active,
    int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ decision_step_idx,
    int64_t* __restrict__ next_action_id,
    int64_t* __restrict__ current_latency_raw_frames,
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
    int64_t* __restrict__ decision_queued_actions,
    int64_t* __restrict__ decision_latency_raw_frames,
    int64_t* __restrict__ decision_ready_raw_frames,
    bool* __restrict__ decision_timeout,
    double* __restrict__ packed_decision,
    float* __restrict__ decision_rewards,
    bool* __restrict__ decision_terminated_out,
    bool* __restrict__ decision_truncated_out,
    int64_t* __restrict__ episode_step_idx,
    const double* __restrict__ reset_upper_pipe_xy,
    const double* __restrict__ reset_lower_pipe_xy,
    int64_t* __restrict__ latency_rng_state,
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    const int64_t* __restrict__ profile_latency_raw_frames,
    int64_t use_profile_latency,
    int64_t obs_stride_raw_frames,
    int64_t hold_duration,
    int64_t ordering_policy,
    int64_t max_episode_steps,
    int64_t reset_pipes_in_kernel,
    int64_t reset_pipe_slot_capacity,
    int batch_size,
    int queue_capacity,
    int base_shift) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }

    double reward_sum = 0.0;
    bool terminated = false;
    bool truncated = false;
    bool offscreen0 = false;
    bool offscreen1 = false;
    bool offscreen2 = false;
    bool timeout = false;
    int64_t reset_slot0 = 0;
    int64_t reset_slot1 = 0;
    int64_t reset_slot2 = 0;
    int64_t raw_frames_advanced = 0;

    if (active[env]) {
        const int64_t latency = use_profile_latency != 0
            ? profile_latency_raw_frames[env]
            : latency_distribution_type == 3
            ? current_latency_raw_frames[env]
            : sample_latency_raw_frames(
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

        const int64_t base = static_cast<int64_t>(env) * queue_capacity;
        const int64_t tail = (pending_head[env] + pending_count[env]) % queue_capacity;
        pending_actions[base + tail] = actions[env];
        pending_ready_raw_frames[base + tail] = ready_raw_frame;
        pending_action_ids[base + tail] = next_action_id[env];
        pending_count[env] += 1;
        next_action_id[env] += 1;

        bool still_active = true;
        for (int64_t raw_step = 0; raw_step < obs_stride_raw_frames && still_active; ++raw_step) {
            int64_t raw_action = 0;
            bool selected = false;
            int64_t selected_action = 0;
            const int64_t count = pending_count[env];
            const int64_t head = pending_head[env];
            const int64_t current_step = env_step_idx[env];

            if (count > 0) {
                if (ordering_policy == 0) {
                    const int64_t pos = head;
                    if (pending_ready_raw_frames[base + pos] <= current_step) {
                        selected = true;
                        selected_action = pending_actions[base + pos];
                        pending_head[env] = (head + 1) % queue_capacity;
                        pending_count[env] = count - 1;
                    }
                } else {
                    int64_t selected_offset = -1;
                    int64_t keep_count = 0;
                    for (int64_t offset = 0; offset < count; ++offset) {
                        const int64_t pos = (head + offset) % queue_capacity;
                        const bool ready = pending_ready_raw_frames[base + pos] <= current_step;
                        if (ready) {
                            selected_offset = offset;
                        } else {
                            scratch_actions[base + keep_count] = pending_actions[base + pos];
                            scratch_ready_raw_frames[base + keep_count] = pending_ready_raw_frames[base + pos];
                            scratch_action_ids[base + keep_count] = pending_action_ids[base + pos];
                            keep_count += 1;
                        }
                    }
                    if (selected_offset >= 0) {
                        const int64_t selected_pos = (head + selected_offset) % queue_capacity;
                        selected = true;
                        selected_action = pending_actions[base + selected_pos];
                        for (int64_t offset = 0; offset < keep_count; ++offset) {
                            pending_actions[base + offset] = scratch_actions[base + offset];
                            pending_ready_raw_frames[base + offset] = scratch_ready_raw_frames[base + offset];
                            pending_action_ids[base + offset] = scratch_action_ids[base + offset];
                        }
                        pending_head[env] = 0;
                        pending_count[env] = keep_count;
                    }
                }
            }

            if (selected) {
                current_applied_action[env] = selected_action;
                persistence_remaining[env] = hold_duration;
            }
            if (persistence_remaining[env] > 0) {
                raw_action = current_applied_action[env];
                persistence_remaining[env] -= 1;
            }
            current_applied_action[env] = raw_action;

            double reward = 0.1;
            const bool flap = (
                raw_action == 1 &&
                player_y[env] > -2.0 * static_cast<double>(PLAYER_HEIGHT));
            if (flap) {
                player_vel_y[env] = static_cast<double>(PLAYER_FLAP_ACC);
            }

            const double player_mid_pos = player_x[env] + static_cast<double>(PLAYER_WIDTH) / 2.0;
            bool scored = false;
            for (int pipe = 0; pipe < 3; ++pipe) {
                const int xy_idx = (env * 3 + pipe) * 2;
                const double pipe_mid_pos = upper_pipe_xy[xy_idx + 0] + static_cast<double>(PIPE_WIDTH) / 2.0;
                const bool passed = (
                    pipe_mid_pos <= player_mid_pos &&
                    player_mid_pos < pipe_mid_pos + 4.0);
                if (passed) {
                    score[env] += 1;
                    scored = true;
                }
            }
            if (scored) {
                reward = 1.0;
            }

            if (((loop_iter[env] + 1) % 3) == 0) {
                const int64_t cycle_pos = player_idx_cycle_pos[env];
                int64_t next_idx = 0;
                if (cycle_pos == 0) {
                    next_idx = 0;
                } else if (cycle_pos == 1) {
                    next_idx = 1;
                } else if (cycle_pos == 2) {
                    next_idx = 2;
                } else {
                    next_idx = 1;
                }
                player_idx[env] = next_idx;
                player_idx_cycle_pos[env] = (cycle_pos + 1) % 4;
            }

            loop_iter[env] = (loop_iter[env] + 1) % 30;
            ground_x[env] = -fmod((-ground_x[env] + 100.0), static_cast<double>(base_shift));

            if (player_rot[env] > -90) {
                player_rot[env] -= PLAYER_VEL_ROT;
            }

            if (!flap && player_vel_y[env] < static_cast<double>(PLAYER_MAX_VEL_Y)) {
                player_vel_y[env] += static_cast<double>(PLAYER_ACC_Y);
            }
            if (flap) {
                player_rot[env] = 45;
            }

            const double max_y_delta = ground_y[env] - player_y[env] - static_cast<double>(PLAYER_HEIGHT);
            const double delta = player_vel_y[env] < max_y_delta ? player_vel_y[env] : max_y_delta;
            player_y[env] += delta;

            for (int pipe = 0; pipe < 3; ++pipe) {
                const int xy_idx = (env * 3 + pipe) * 2;
                upper_pipe_xy[xy_idx + 0] += static_cast<double>(PIPE_VEL_X);
                lower_pipe_xy[xy_idx + 0] += static_cast<double>(PIPE_VEL_X);
                const bool offscreen = upper_pipe_xy[xy_idx + 0] < -static_cast<double>(PIPE_WIDTH);
                if (pipe == 0) {
                    offscreen0 = offscreen0 || offscreen;
                } else if (pipe == 1) {
                    offscreen1 = offscreen1 || offscreen;
                } else {
                    offscreen2 = offscreen2 || offscreen;
                }
                if (offscreen && reset_pipes_in_kernel != 0) {
                    int64_t reset_slot = 0;
                    if (pipe == 0) {
                        reset_slot = reset_slot0;
                        reset_slot0 += 1;
                    } else if (pipe == 1) {
                        reset_slot = reset_slot1;
                        reset_slot1 += 1;
                    } else {
                        reset_slot = reset_slot2;
                        reset_slot2 += 1;
                    }
                    const int reset_idx = (((env * 3 + pipe) * reset_pipe_slot_capacity + reset_slot) * 2);
                    upper_pipe_xy[xy_idx + 0] = reset_upper_pipe_xy[reset_idx + 0];
                    upper_pipe_xy[xy_idx + 1] = reset_upper_pipe_xy[reset_idx + 1];
                    lower_pipe_xy[xy_idx + 0] = reset_lower_pipe_xy[reset_idx + 0];
                    lower_pipe_xy[xy_idx + 1] = reset_lower_pipe_xy[reset_idx + 1];
                }
            }

            if (player_y[env] < 0.0) {
                reward = -0.5;
            }

            const bool ground = player_y[env] + static_cast<double>(PLAYER_HEIGHT) >= ground_y[env] - 1.0;
            bool crashed = ground;
            if (!ground) {
                const int64_t px = static_cast<int64_t>(player_x[env]);
                const int64_t py = static_cast<int64_t>(player_y[env]);
                for (int pipe = 0; pipe < 3; ++pipe) {
                    const int xy_idx = (env * 3 + pipe) * 2;
                    const int64_t up_x = static_cast<int64_t>(upper_pipe_xy[xy_idx + 0]);
                    const int64_t up_y = static_cast<int64_t>(upper_pipe_xy[xy_idx + 1]);
                    const int64_t low_x = static_cast<int64_t>(lower_pipe_xy[xy_idx + 0]);
                    const int64_t low_y = static_cast<int64_t>(lower_pipe_xy[xy_idx + 1]);
                    crashed = crashed || rects_collide(px, py, up_x, up_y) || rects_collide(px, py, low_x, low_y);
                }
            }
            if (crashed) {
                reward = -1.0;
                terminated = true;
                player_vel_y[env] = 0.0;
            }

            if (score_limit[env] >= 0 && score[env] >= score_limit[env]) {
                truncated = true;
            }

            reward_sum += reward;
            raw_frames_advanced += 1;
            env_step_idx[env] += 1;
            episode_step_idx[env] += 1;
            if (max_episode_steps > 0 && episode_step_idx[env] >= max_episode_steps) {
                truncated = true;
                timeout = true;
            }
            still_active = !(terminated || truncated);
        }
        decision_step_idx[env] += 1;
    }

    const int packed_idx = env * 9;
    packed_decision[packed_idx + 0] = reward_sum;
    packed_decision[packed_idx + 1] = terminated ? 1.0 : 0.0;
    packed_decision[packed_idx + 2] = truncated ? 1.0 : 0.0;
    packed_decision[packed_idx + 3] = static_cast<double>(score[env]);
    packed_decision[packed_idx + 4] = static_cast<double>(raw_frames_advanced);
    packed_decision[packed_idx + 5] = offscreen0 ? 1.0 : 0.0;
    packed_decision[packed_idx + 6] = offscreen1 ? 1.0 : 0.0;
    packed_decision[packed_idx + 7] = offscreen2 ? 1.0 : 0.0;
    packed_decision[packed_idx + 8] = static_cast<double>(decision_latency_raw_frames[env]);
    decision_rewards[env] = static_cast<float>(reward_sum);
    decision_terminated_out[env] = terminated;
    decision_truncated_out[env] = truncated;
    decision_timeout[env] = timeout;
}

__global__ void flappy_latency_reset_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ pending_head,
    int64_t* __restrict__ pending_count,
    int64_t* __restrict__ current_applied_action,
    int64_t* __restrict__ env_step_idx,
    int64_t* __restrict__ decision_step_idx,
    int64_t* __restrict__ next_action_id,
    int64_t* __restrict__ persistence_remaining,
    int64_t* __restrict__ current_latency_raw_frames,
    int64_t* __restrict__ episode_idx,
    int64_t fixed_latency_raw_frames,
    int count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }
    const int64_t env = indices[item];
    pending_head[env] = 0;
    pending_count[env] = 0;
    current_applied_action[env] = 0;
    env_step_idx[env] = 0;
    decision_step_idx[env] = 0;
    next_action_id[env] = 0;
    persistence_remaining[env] = 0;
    current_latency_raw_frames[env] = fixed_latency_raw_frames;
    episode_idx[env] += 1;
}

__global__ void flappy_reset_kernel(
    const int64_t* __restrict__ indices,
    const double* __restrict__ upper_pipe_xy_reset,
    const double* __restrict__ lower_pipe_xy_reset,
    double* __restrict__ player_x,
    double* __restrict__ player_y,
    double* __restrict__ player_vel_y,
    int64_t* __restrict__ player_rot,
    int64_t* __restrict__ player_idx,
    int64_t* __restrict__ player_idx_cycle_pos,
    int64_t* __restrict__ loop_iter,
    int64_t* __restrict__ score,
    double* __restrict__ ground_x,
    double* __restrict__ ground_y,
    double* __restrict__ upper_pipe_xy,
    double* __restrict__ lower_pipe_xy,
    int count,
    int screen_width,
    int screen_height) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }
    const int64_t env = indices[item];
    player_x[env] = static_cast<double>(static_cast<int>(screen_width * 0.2));
    player_y[env] = static_cast<double>(static_cast<int>((screen_height - PLAYER_HEIGHT) / 2));
    player_vel_y[env] = -9.0;
    player_rot[env] = 45;
    player_idx[env] = 0;
    player_idx_cycle_pos[env] = 0;
    loop_iter[env] = 0;
    score[env] = 0;
    ground_x[env] = 0.0;
    ground_y[env] = static_cast<double>(screen_height) * 0.79;

    for (int pipe = 0; pipe < 3; ++pipe) {
        const int64_t env_xy = (env * 3 + pipe) * 2;
        const int64_t reset_xy = (static_cast<int64_t>(item) * 3 + pipe) * 2;
        upper_pipe_xy[env_xy + 0] = upper_pipe_xy_reset[reset_xy + 0];
        upper_pipe_xy[env_xy + 1] = upper_pipe_xy_reset[reset_xy + 1];
        lower_pipe_xy[env_xy + 0] = lower_pipe_xy_reset[reset_xy + 0];
        lower_pipe_xy[env_xy + 1] = lower_pipe_xy_reset[reset_xy + 1];
    }
}

}  // namespace

void flappy_step_cuda(
    torch::Tensor player_x,
    torch::Tensor player_y,
    torch::Tensor player_vel_y,
    torch::Tensor player_rot,
    torch::Tensor player_idx,
    torch::Tensor player_idx_cycle_pos,
    torch::Tensor loop_iter,
    torch::Tensor score,
    torch::Tensor score_limit,
    torch::Tensor ground_x,
    torch::Tensor ground_y,
    torch::Tensor upper_pipe_xy,
    torch::Tensor lower_pipe_xy,
    torch::Tensor actions,
    torch::Tensor active,
    torch::Tensor rewards,
    torch::Tensor terminated,
    torch::Tensor truncated,
    torch::Tensor offscreen,
    torch::Tensor packed_step,
    int64_t batch_size,
    int64_t screen_width,
    int64_t screen_height,
    int64_t base_shift) {
    (void)screen_width;
    (void)screen_height;
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_step_kernel<<<blocks, threads>>>(
        player_x.data_ptr<double>(),
        player_y.data_ptr<double>(),
        player_vel_y.data_ptr<double>(),
        player_rot.data_ptr<int64_t>(),
        player_idx.data_ptr<int64_t>(),
        player_idx_cycle_pos.data_ptr<int64_t>(),
        loop_iter.data_ptr<int64_t>(),
        score.data_ptr<int64_t>(),
        score_limit.data_ptr<int64_t>(),
        ground_x.data_ptr<double>(),
        ground_y.data_ptr<double>(),
        upper_pipe_xy.data_ptr<double>(),
        lower_pipe_xy.data_ptr<double>(),
        actions.data_ptr<int64_t>(),
        active.data_ptr<bool>(),
        rewards.data_ptr<double>(),
        terminated.data_ptr<bool>(),
        truncated.data_ptr<bool>(),
        offscreen.data_ptr<bool>(),
        packed_step.data_ptr<double>(),
        static_cast<int>(batch_size),
        static_cast<int>(base_shift));
}

void flappy_render_cuda(
    torch::Tensor out,
    torch::Tensor background_rgb,
    torch::Tensor fill_color,
    bool has_background,
    torch::Tensor base_rgb,
    torch::Tensor base_mask,
    torch::Tensor pipe_upper_rgb,
    torch::Tensor pipe_upper_mask,
    torch::Tensor pipe_lower_rgb,
    torch::Tensor pipe_lower_mask,
    torch::Tensor bird_rgb,
    torch::Tensor bird_mask,
    torch::Tensor bird_heights,
    torch::Tensor bird_widths,
    torch::Tensor player_x,
    torch::Tensor player_y,
    torch::Tensor player_idx,
    torch::Tensor player_rot,
    torch::Tensor ground_x,
    torch::Tensor ground_y,
    torch::Tensor upper_pipe_xy,
    torch::Tensor lower_pipe_xy,
    int64_t batch_size,
    int64_t screen_width,
    int64_t screen_height,
    int64_t num_rotations,
    int64_t min_rotation) {
    const int64_t total_pixels = static_cast<int64_t>(batch_size) * screen_width * screen_height;
    const int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1) / threads);
    flappy_render_kernel<<<blocks, threads>>>(
        out.data_ptr<unsigned char>(),
        has_background ? background_rgb.data_ptr<unsigned char>() : nullptr,
        fill_color.data_ptr<unsigned char>(),
        has_background,
        base_rgb.data_ptr<unsigned char>(),
        base_mask.data_ptr<bool>(),
        static_cast<int>(base_rgb.size(0)),
        static_cast<int>(base_rgb.size(1)),
        pipe_upper_rgb.data_ptr<unsigned char>(),
        pipe_upper_mask.data_ptr<bool>(),
        static_cast<int>(pipe_upper_rgb.size(0)),
        static_cast<int>(pipe_upper_rgb.size(1)),
        pipe_lower_rgb.data_ptr<unsigned char>(),
        pipe_lower_mask.data_ptr<bool>(),
        static_cast<int>(pipe_lower_rgb.size(0)),
        static_cast<int>(pipe_lower_rgb.size(1)),
        bird_rgb.data_ptr<unsigned char>(),
        bird_mask.data_ptr<bool>(),
        bird_heights.data_ptr<int64_t>(),
        bird_widths.data_ptr<int64_t>(),
        static_cast<int>(bird_rgb.size(1)),
        static_cast<int>(bird_rgb.size(2)),
        player_x.data_ptr<double>(),
        player_y.data_ptr<double>(),
        player_idx.data_ptr<int64_t>(),
        player_rot.data_ptr<int64_t>(),
        ground_x.data_ptr<double>(),
        ground_y.data_ptr<double>(),
        upper_pipe_xy.data_ptr<double>(),
        lower_pipe_xy.data_ptr<double>(),
        static_cast<int>(batch_size),
        static_cast<int>(screen_width),
        static_cast<int>(screen_height),
        static_cast<int>(num_rotations),
        static_cast<int>(min_rotation),
        total_pixels);
}

void flappy_latency_begin_cuda(
    torch::Tensor actions,
    torch::Tensor active,
    torch::Tensor env_step_idx,
    torch::Tensor next_action_id,
    torch::Tensor current_latency_raw_frames,
    torch::Tensor pending_actions,
    torch::Tensor pending_ready_raw_frames,
    torch::Tensor pending_action_ids,
    torch::Tensor pending_head,
    torch::Tensor pending_count,
    torch::Tensor decision_queued_actions,
    torch::Tensor decision_latency_raw_frames,
    torch::Tensor decision_ready_raw_frames,
    torch::Tensor latency_rng_state,
    torch::Tensor profile_latency_raw_frames,
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    int64_t batch_size,
    int64_t queue_capacity) {
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_latency_begin_kernel<<<blocks, threads>>>(
        actions.data_ptr<int64_t>(),
        active.data_ptr<bool>(),
        env_step_idx.data_ptr<int64_t>(),
        next_action_id.data_ptr<int64_t>(),
        current_latency_raw_frames.data_ptr<int64_t>(),
        pending_actions.data_ptr<int64_t>(),
        pending_ready_raw_frames.data_ptr<int64_t>(),
        pending_action_ids.data_ptr<int64_t>(),
        pending_head.data_ptr<int64_t>(),
        pending_count.data_ptr<int64_t>(),
        decision_queued_actions.data_ptr<int64_t>(),
        decision_latency_raw_frames.data_ptr<int64_t>(),
        decision_ready_raw_frames.data_ptr<int64_t>(),
        latency_rng_state.data_ptr<int64_t>(),
        fixed_latency_raw_frames,
        latency_distribution_type,
        latency_distribution_mean,
        latency_distribution_std,
        latency_distribution_min,
        latency_distribution_max,
        profile_latency_raw_frames.data_ptr<int64_t>(),
        latency_distribution_type == 4 ? 1 : 0,
        static_cast<int>(batch_size),
        static_cast<int>(queue_capacity));
}

void flappy_latency_raw_actions_cuda(
    torch::Tensor active,
    torch::Tensor env_step_idx,
    torch::Tensor current_applied_action,
    torch::Tensor persistence_remaining,
    torch::Tensor pending_actions,
    torch::Tensor pending_ready_raw_frames,
    torch::Tensor pending_action_ids,
    torch::Tensor pending_head,
    torch::Tensor pending_count,
    torch::Tensor scratch_actions,
    torch::Tensor scratch_ready_raw_frames,
    torch::Tensor scratch_action_ids,
    torch::Tensor raw_actions,
    int64_t hold_duration,
    int64_t ordering_policy,
    int64_t batch_size,
    int64_t queue_capacity) {
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_latency_raw_actions_kernel<<<blocks, threads>>>(
        active.data_ptr<bool>(),
        env_step_idx.data_ptr<int64_t>(),
        current_applied_action.data_ptr<int64_t>(),
        persistence_remaining.data_ptr<int64_t>(),
        pending_actions.data_ptr<int64_t>(),
        pending_ready_raw_frames.data_ptr<int64_t>(),
        pending_action_ids.data_ptr<int64_t>(),
        pending_head.data_ptr<int64_t>(),
        pending_count.data_ptr<int64_t>(),
        scratch_actions.data_ptr<int64_t>(),
        scratch_ready_raw_frames.data_ptr<int64_t>(),
        scratch_action_ids.data_ptr<int64_t>(),
        raw_actions.data_ptr<int64_t>(),
        hold_duration,
        ordering_policy,
        static_cast<int>(batch_size),
        static_cast<int>(queue_capacity));
}

void flappy_latency_record_cuda(
    torch::Tensor advanced,
    torch::Tensor env_step_idx,
    int64_t batch_size) {
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_latency_record_kernel<<<blocks, threads>>>(
        advanced.data_ptr<bool>(),
        env_step_idx.data_ptr<int64_t>(),
        static_cast<int>(batch_size));
}

void flappy_latency_finish_cuda(
    torch::Tensor active,
    torch::Tensor decision_step_idx,
    int64_t batch_size) {
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_latency_finish_kernel<<<blocks, threads>>>(
        active.data_ptr<bool>(),
        decision_step_idx.data_ptr<int64_t>(),
        static_cast<int>(batch_size));
}

void flappy_latency_reset_cuda(
    torch::Tensor indices,
    torch::Tensor pending_head,
    torch::Tensor pending_count,
    torch::Tensor current_applied_action,
    torch::Tensor env_step_idx,
    torch::Tensor decision_step_idx,
    torch::Tensor next_action_id,
    torch::Tensor persistence_remaining,
    torch::Tensor current_latency_raw_frames,
    torch::Tensor episode_idx,
    int64_t fixed_latency_raw_frames,
    int64_t count) {
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    flappy_latency_reset_kernel<<<blocks, threads>>>(
        indices.data_ptr<int64_t>(),
        pending_head.data_ptr<int64_t>(),
        pending_count.data_ptr<int64_t>(),
        current_applied_action.data_ptr<int64_t>(),
        env_step_idx.data_ptr<int64_t>(),
        decision_step_idx.data_ptr<int64_t>(),
        next_action_id.data_ptr<int64_t>(),
        persistence_remaining.data_ptr<int64_t>(),
        current_latency_raw_frames.data_ptr<int64_t>(),
        episode_idx.data_ptr<int64_t>(),
        fixed_latency_raw_frames,
        static_cast<int>(count));
}

void flappy_latency_core_decision_cuda(
    torch::Tensor player_x,
    torch::Tensor player_y,
    torch::Tensor player_vel_y,
    torch::Tensor player_rot,
    torch::Tensor player_idx,
    torch::Tensor player_idx_cycle_pos,
    torch::Tensor loop_iter,
    torch::Tensor score,
    torch::Tensor score_limit,
    torch::Tensor ground_x,
    torch::Tensor ground_y,
    torch::Tensor upper_pipe_xy,
    torch::Tensor lower_pipe_xy,
    torch::Tensor actions,
    torch::Tensor active,
    torch::Tensor env_step_idx,
    torch::Tensor decision_step_idx,
    torch::Tensor next_action_id,
    torch::Tensor current_latency_raw_frames,
    torch::Tensor current_applied_action,
    torch::Tensor persistence_remaining,
    torch::Tensor pending_actions,
    torch::Tensor pending_ready_raw_frames,
    torch::Tensor pending_action_ids,
    torch::Tensor pending_head,
    torch::Tensor pending_count,
    torch::Tensor scratch_actions,
    torch::Tensor scratch_ready_raw_frames,
    torch::Tensor scratch_action_ids,
    torch::Tensor decision_queued_actions,
    torch::Tensor decision_latency_raw_frames,
    torch::Tensor decision_ready_raw_frames,
    torch::Tensor decision_timeout,
    torch::Tensor packed_decision,
    torch::Tensor decision_rewards,
    torch::Tensor decision_terminated,
    torch::Tensor decision_truncated,
    torch::Tensor episode_step_idx,
    torch::Tensor reset_upper_pipe_xy,
    torch::Tensor reset_lower_pipe_xy,
    torch::Tensor latency_rng_state,
    torch::Tensor profile_latency_raw_frames,
    int64_t fixed_latency_raw_frames,
    int64_t latency_distribution_type,
    double latency_distribution_mean,
    double latency_distribution_std,
    double latency_distribution_min,
    double latency_distribution_max,
    int64_t obs_stride_raw_frames,
    int64_t hold_duration,
    int64_t ordering_policy,
    int64_t max_episode_steps,
    int64_t reset_pipes_in_kernel,
    int64_t reset_pipe_slot_capacity,
    int64_t batch_size,
    int64_t queue_capacity,
    int64_t base_shift) {
    const int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    flappy_latency_core_decision_kernel<<<blocks, threads>>>(
        player_x.data_ptr<double>(),
        player_y.data_ptr<double>(),
        player_vel_y.data_ptr<double>(),
        player_rot.data_ptr<int64_t>(),
        player_idx.data_ptr<int64_t>(),
        player_idx_cycle_pos.data_ptr<int64_t>(),
        loop_iter.data_ptr<int64_t>(),
        score.data_ptr<int64_t>(),
        score_limit.data_ptr<int64_t>(),
        ground_x.data_ptr<double>(),
        ground_y.data_ptr<double>(),
        upper_pipe_xy.data_ptr<double>(),
        lower_pipe_xy.data_ptr<double>(),
        actions.data_ptr<int64_t>(),
        active.data_ptr<bool>(),
        env_step_idx.data_ptr<int64_t>(),
        decision_step_idx.data_ptr<int64_t>(),
        next_action_id.data_ptr<int64_t>(),
        current_latency_raw_frames.data_ptr<int64_t>(),
        current_applied_action.data_ptr<int64_t>(),
        persistence_remaining.data_ptr<int64_t>(),
        pending_actions.data_ptr<int64_t>(),
        pending_ready_raw_frames.data_ptr<int64_t>(),
        pending_action_ids.data_ptr<int64_t>(),
        pending_head.data_ptr<int64_t>(),
        pending_count.data_ptr<int64_t>(),
        scratch_actions.data_ptr<int64_t>(),
        scratch_ready_raw_frames.data_ptr<int64_t>(),
        scratch_action_ids.data_ptr<int64_t>(),
        decision_queued_actions.data_ptr<int64_t>(),
        decision_latency_raw_frames.data_ptr<int64_t>(),
        decision_ready_raw_frames.data_ptr<int64_t>(),
        decision_timeout.data_ptr<bool>(),
        packed_decision.data_ptr<double>(),
        decision_rewards.data_ptr<float>(),
        decision_terminated.data_ptr<bool>(),
        decision_truncated.data_ptr<bool>(),
        episode_step_idx.data_ptr<int64_t>(),
        reset_upper_pipe_xy.data_ptr<double>(),
        reset_lower_pipe_xy.data_ptr<double>(),
        latency_rng_state.data_ptr<int64_t>(),
        fixed_latency_raw_frames,
        latency_distribution_type,
        latency_distribution_mean,
        latency_distribution_std,
        latency_distribution_min,
        latency_distribution_max,
        profile_latency_raw_frames.data_ptr<int64_t>(),
        latency_distribution_type == 4 ? 1 : 0,
        obs_stride_raw_frames,
        hold_duration,
        ordering_policy,
        max_episode_steps,
        reset_pipes_in_kernel,
        reset_pipe_slot_capacity,
        static_cast<int>(batch_size),
        static_cast<int>(queue_capacity),
        static_cast<int>(base_shift));
}

void flappy_reset_cuda(
    torch::Tensor indices,
    torch::Tensor upper_pipe_xy_reset,
    torch::Tensor lower_pipe_xy_reset,
    torch::Tensor player_x,
    torch::Tensor player_y,
    torch::Tensor player_vel_y,
    torch::Tensor player_rot,
    torch::Tensor player_idx,
    torch::Tensor player_idx_cycle_pos,
    torch::Tensor loop_iter,
    torch::Tensor score,
    torch::Tensor ground_x,
    torch::Tensor ground_y,
    torch::Tensor upper_pipe_xy,
    torch::Tensor lower_pipe_xy,
    int64_t count,
    int64_t screen_width,
    int64_t screen_height) {
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    flappy_reset_kernel<<<blocks, threads>>>(
        indices.data_ptr<int64_t>(),
        upper_pipe_xy_reset.data_ptr<double>(),
        lower_pipe_xy_reset.data_ptr<double>(),
        player_x.data_ptr<double>(),
        player_y.data_ptr<double>(),
        player_vel_y.data_ptr<double>(),
        player_rot.data_ptr<int64_t>(),
        player_idx.data_ptr<int64_t>(),
        player_idx_cycle_pos.data_ptr<int64_t>(),
        loop_iter.data_ptr<int64_t>(),
        score.data_ptr<int64_t>(),
        ground_x.data_ptr<double>(),
        ground_y.data_ptr<double>(),
        upper_pipe_xy.data_ptr<double>(),
        lower_pipe_xy.data_ptr<double>(),
        static_cast<int>(count),
        static_cast<int>(screen_width),
        static_cast<int>(screen_height));
}
