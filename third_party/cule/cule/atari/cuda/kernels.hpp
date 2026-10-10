#pragma once

#include <cule/config.hpp>

#include <cule/atari/state.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/prng.hpp>

#include <cule/atari/cuda/frame_state.hpp>
#include <cule/atari/cuda/state.hpp>

#include <agency/agency.hpp>
#include <agency/cuda.hpp>

namespace cule
{
namespace atari
{
namespace cuda
{

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void initialize_frame_states_kernel(const uint32_t noop_reset_steps,
                                    State_t* cached_states_buffer,
                                    frame_state* cached_frame_states_buffer,
                                    const uint32_t* pf_base,
                                    const uint8_t* p0_base,
                                    const uint8_t* p1_base,
                                    const uint8_t* m0_base,
                                    const uint8_t* m1_base,
                                    const uint8_t* bl_base)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= noop_reset_steps)
    {
        return;
    }

    frame_state& fs  = cached_frame_states_buffer[global_index];
    fs.CurrentPFMask = &playfield_accessor(0, 0) + (fs.CurrentPFMask - pf_base);
    fs.CurrentP0Mask = &player_mask_accessor(0, 0, 0, 0) + (fs.CurrentP0Mask - p0_base);
    fs.CurrentP1Mask = &player_mask_accessor(0, 0, 0, 0) + (fs.CurrentP1Mask - p1_base);
    fs.CurrentM0Mask = &missle_accessor(0, 0, 0, 0) + (fs.CurrentM0Mask - m0_base);
    fs.CurrentM1Mask = &missle_accessor(0, 0, 0, 0) + (fs.CurrentM1Mask - m1_base);
    fs.CurrentBLMask = &ball_accessor(0, 0, 0) + (fs.CurrentBLMask - bl_base);

    State_t& s = cached_states_buffer[global_index];
    s.CurrentPFMask = fs.CurrentPFMask;
    s.CurrentP0Mask = fs.CurrentP0Mask;
    s.CurrentP1Mask = fs.CurrentP1Mask;
    s.CurrentM0Mask = fs.CurrentM0Mask;
    s.CurrentM1Mask = fs.CurrentM1Mask;
    s.CurrentBLMask = fs.CurrentBLMask;
}

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void initialize_states_kernel(const uint32_t num_envs,
                              const uint32_t noop_reset_steps,
                              State_t* states_buffer,
                              uint8_t* ram_buffer,
                              const State_t* cached_states_buffer,
                              const uint8_t* cached_ram_buffer,
                              uint32_t* rand_states_buffer,
                              frame_state* frame_states_buffer,
                              frame_state* cached_frame_states_buffer,
                              uint8_t* frame_buffer,
                              const uint8_t* cached_frame_buffer)
{
    enum
    {
        NUM_INT_REGS = 128 / sizeof(uint32_t),
    };

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    prng gen(rand_states_buffer[global_index]);
    const size_t cache_index = gen.sample() % noop_reset_steps;

    uint32_t ram[NUM_INT_REGS];

    states_buffer[global_index] = cached_states_buffer[cache_index];
    frame_states_buffer[global_index] = cached_frame_states_buffer[cache_index];

    uint32_t * ram_int = (uint32_t*) cached_ram_buffer + (NUM_INT_REGS * cache_index);

    #pragma loop unroll
    for(int32_t i = 0; i < NUM_INT_REGS; i++)
    {
        ram[i] = ram_int[i];
    }

    ram_int = ((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x;

    #pragma loop unroll
    for(int32_t i = 0; i < NUM_INT_REGS; i++)
    {
        ram_int[i * NT] = ram[i];
    }
    for(size_t i = 0; i < 300 * SCREEN_WIDTH; ++i)
    {
        frame_buffer[global_index * 300 * SCREEN_WIDTH + i] =
            cached_frame_buffer[cache_index * 300 * SCREEN_WIDTH + i];
    }
}

template<typename State_t, typename Environment_t, size_t NT>
__launch_bounds__(NT) __global__
void step_kernel(const uint32_t num_envs,
                 const bool fire_reset,
                 State_t* states_buffer,
                 uint8_t* ram_buffer,
                 uint32_t* tia_update_buffer,
                 const Action* player_a_buffer,
                 const Action* player_b_buffer,
                 bool* done_buffer)
{
    enum
    {
        NUM_INT_REGS = 128 / sizeof(uint32_t),
    };

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if((global_index >= num_envs) || done_buffer[global_index])
    {
        return;
    }

    states_buffer += global_index;

    uint32_t ram[NUM_INT_REGS];

    State_t s;

    {
        state_store_load_helper(s, *states_buffer);

        uint32_t * ram_int = ((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x;

        #pragma loop unroll
        for(int32_t i = 0; i < NUM_INT_REGS; i++)
        {
            ram[i] = ram_int[i * NT];
        }
    }

    Action player_a_action = ACTION_NOOP;
    Action player_b_action = ACTION_NOOP;

    if(player_a_buffer != nullptr)
    {
        player_a_action = player_a_buffer[global_index];
    }
    if(player_b_buffer != nullptr)
    {
        player_b_action = player_b_buffer[global_index];
    }

    if(fire_reset && s.tiaFlags[FLAG_ALE_LOST_LIFE])
    {
        player_a_action = ACTION_FIRE;
        s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);
    }

    s.ram = ram;
    s.rom = gpu_rom;
    s.tia_update_buffer = tia_update_buffer + (ENV_UPDATE_SIZE * global_index);

    Environment_t::act(s, player_a_action, player_b_action);

    {
        state_store_load_helper(*states_buffer, s);

        uint32_t * ram_int = ((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x;

        #pragma loop unroll
        for(int32_t i = 0; i < NUM_INT_REGS; i++)
        {
            ram_int[i * NT] = ram[i];
        }
    }
}

template<typename State_t, typename Environment_t, size_t NT>
__launch_bounds__(NT) __global__
void step_render_kernel(const uint32_t num_envs,
                        const bool fire_reset,
                        State_t* states_buffer,
                        uint8_t* ram_buffer,
                        uint32_t* tia_update_buffer,
                        frame_state* frame_states_buffer,
                        uint8_t* frame_buffer,
                        const Action* player_a_buffer,
                        const Action* player_b_buffer,
                        bool* done_buffer)
{
    enum
    {
        NUM_INT_REGS = 128 / sizeof(uint32_t),
    };

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if((global_index >= num_envs) || done_buffer[global_index])
    {
        return;
    }

    states_buffer += global_index;
    frame_states_buffer += global_index;

    uint32_t ram[NUM_INT_REGS];

    State_t s;
    frame_state fs;

    {
        state_store_load_helper(s, *states_buffer);
        state_store_load_helper(fs, *frame_states_buffer);

        s.Color = fs.Color;
        s.playfieldPriorityAndScore = fs.playfieldPriorityAndScore;

        uint32_t * ram_int = ((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x;

        #pragma loop unroll
        for(int32_t i = 0; i < NUM_INT_REGS; i++)
        {
            ram[i] = ram_int[i * NT];
        }
    }

    Action player_a_action = ACTION_NOOP;
    Action player_b_action = ACTION_NOOP;

    if(player_a_buffer != nullptr)
    {
        player_a_action = player_a_buffer[global_index];
    }
    if(player_b_buffer != nullptr)
    {
        player_b_action = player_b_buffer[global_index];
    }

    if(fire_reset && s.tiaFlags[FLAG_ALE_LOST_LIFE])
    {
        player_a_action = ACTION_FIRE;
        s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);
    }

    s.ram = ram;
    s.rom = gpu_rom;
    s.tia_update_buffer = tia_update_buffer + (ENV_UPDATE_SIZE * global_index);
    s.framePointer = frame_buffer + (global_index * 300 * SCREEN_WIDTH);

    Environment_t::act(s, player_a_action, player_b_action);

    {
        state_store_load_helper(*states_buffer, s);

        fs.Color = s.Color;
        fs.GRP = s.GRP;
        fs.HM = s.HM;
        fs.PF = s.PF;
        fs.POS = s.POS;
        fs.CurrentGRP0 = s.CurrentGRP0;
        fs.CurrentGRP1 = s.CurrentGRP1;
        fs.clockWhenFrameStarted = s.clockWhenFrameStarted;
        fs.clockAtLastUpdate = s.clockAtLastUpdate;
        fs.lastHMOVEClock = s.lastHMOVEClock;
        fs.playfieldPriorityAndScore = s.playfieldPriorityAndScore;
        fs.tiaFlags = s.tiaFlags;
        fs.M0CosmicArkCounter = s.M0CosmicArkCounter;
        fs.CurrentPFMask = s.CurrentPFMask;
        fs.CurrentP0Mask = s.CurrentP0Mask;
        fs.CurrentP1Mask = s.CurrentP1Mask;
        fs.CurrentM0Mask = s.CurrentM0Mask;
        fs.CurrentM1Mask = s.CurrentM1Mask;
        fs.CurrentBLMask = s.CurrentBLMask;
        state_store_load_helper(*frame_states_buffer, fs);

        uint32_t * ram_int = ((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x;

        #pragma loop unroll
        for(int32_t i = 0; i < NUM_INT_REGS; i++)
        {
            ram_int[i * NT] = ram[i];
        }
    }
}

template<typename State_t, typename ALE_t, size_t NT>
__launch_bounds__(NT) __global__
void get_data_kernel(const int32_t num_envs,
                     const bool episodic_life,
                     State_t* states_buffer,
                     const uint8_t* ram_buffer,
                     bool* done_buffer,
                     float* rewards_buffer,
                     int32_t* lives_buffer)
{

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if((global_index >= num_envs) || done_buffer[global_index])
    {
        return;
    }

    State_t& s = states_buffer[global_index];
    s.ram = (uint32_t*)(ram_buffer + (128 * global_index));

    const uint32_t old_lives = lives_buffer[global_index];
    const uint32_t new_lives = ALE_t::getLives(s);
    lives_buffer[global_index] = new_lives;

    const bool lost_life = new_lives < old_lives;
    s.tiaFlags.template change<FLAG_ALE_LOST_LIFE>(lost_life);

    rewards_buffer[global_index] += ALE_t::getRewards(s);
    done_buffer[global_index] |= s.tiaFlags[FLAG_ALE_TERMINAL] || (episodic_life && lost_life);

    s.score = ALE_t::getScore(s);
}

template<size_t NT>
__launch_bounds__(NT) __global__
void process_kernel(const uint32_t num_envs,
                    const uint32_t* tia_update_buffer,
                    frame_state* frame_states_buffer,
                    uint8_t* frame_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;
    if(global_index >= num_envs)
    {
        return;
    }

    frame_state fs;
    state_store_load_helper(fs, frame_states_buffer[global_index]);
    fs.srcBuffer = tia_update_buffer + global_index * ENV_UPDATE_SIZE;
    fs.framePointer = frame_buffer + global_index * 300 * SCREEN_WIDTH;
    preprocess::state_to_buffer(fs);
    state_store_load_helper(frame_states_buffer[global_index], fs);
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_palette_kernel(const int32_t num_envs,
                          const int32_t screen_height,
                          const int32_t num_channels,
                          uint8_t* dst_buffer,
                          const uint8_t* src_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < num_envs * screen_height * SCREEN_WIDTH)
    {
        const uint32_t state_index = global_index / (screen_height * SCREEN_WIDTH);

        // slide the start index of the src_buffer forward to account for the
        // mismatch between the number of PAL and NTSC rows
        src_buffer += state_index * SCREEN_WIDTH * (300 - screen_height);

        int32_t color = src_buffer[global_index];
        dst_buffer += num_channels * global_index;

        if(num_channels == 3)
        {
            int32_t rgb = gpu_NTSCPalette[color];
            dst_buffer[0] = uint8_t(rgb >> 16);  // r
            dst_buffer[1] = uint8_t(rgb >>  8);  // g
            dst_buffer[2] = uint8_t(rgb >>  0);  // b
        }
        else
        {
            dst_buffer[0] = uint8_t(gpu_NTSCPalette[color + 1] & 0xFF);
        }
    }
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_rescale_kernel(const int32_t num_envs,
                          const int32_t screen_height,
                          uint8_t * dst_buffer,
                          const uint8_t * src_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < (num_envs * SCALED_SCREEN_SIZE))
    {
        // slide the start index of the src_buffer forward to account for the
        // mismatch between the number of PAL and NTSC rows
        const uint32_t state_index = global_index / SCALED_SCREEN_SIZE;
        src_buffer += state_index * SCREEN_WIDTH * (300 - screen_height);

        const float S_R = float(screen_height) / 84.0f;
        const float S_C = float(SCREEN_WIDTH) / 84.0f;

        const size_t row  = std::floor(float(global_index) / 84.0f);
        const size_t col  = global_index % 84;

        const float rf = (0.5f + row) * S_R;
        const float cf = (0.5f + col) * S_C;
        const size_t r = std::floor(rf - 0.5f);
        const size_t c = std::floor(cf - 0.5f);

        const float delta_R = rf - (0.5f + r);
        const float delta_C = cf - (0.5f + c);

        const float color_0_0 = gpu_NTSCPalette[src_buffer[(r + 0) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_0_1 = gpu_NTSCPalette[src_buffer[(r + 1) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_1_0 = gpu_NTSCPalette[src_buffer[(r + 0) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;
        const float color_1_1 = gpu_NTSCPalette[src_buffer[(r + 1) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;

        const float value = (color_0_0 * (1.0f - delta_R) * (1.0f - delta_C)) +
                            (color_0_1 * delta_R * (1.0f - delta_C)) +
                            (color_1_0 * (1.0f - delta_R) * delta_C) +
                            (color_1_1 * delta_R * delta_C);
        dst_buffer[global_index] = value + 0.5f;
    }
}

} // end namespace cuda
} // end namespace atari
} // end namespace cule
