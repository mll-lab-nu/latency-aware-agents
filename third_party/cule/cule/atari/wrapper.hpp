#pragma once

#include <cule/config.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/state.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{

template<int32_t> struct environment;

class wrapper
{
public:

    using State_t = state;

    wrapper(const rom& cart,
            const size_t num_envs,
            const size_t noop_reset_steps = 30);

    size_t image_buffer_size(const size_t num_channels, const bool rescale) const;

    size_t size() const;

    void initialize_ptrs(State_t* states_ptr,
                         frame_state* frame_states_ptr,
                         uint8_t* ram_ptr,
                         uint32_t* tia_update_ptr,
                         uint8_t* frame_ptr,
                         Action* minimal_actions_ptr,
                         uint32_t* rand_states_ptr,
                         State_t* cached_states_ptr,
                         uint8_t* cached_ram_ptr,
                         frame_state* cached_frame_states_ptr,
                         uint8_t* cached_frame_ptr,
                         uint32_t* cache_index_ptr);

    template<typename ExecutionPolicy>
    void reset(ExecutionPolicy&& policy,
               uint32_t* seedBuffer);

    template<typename ExecutionPolicy>
    void step(ExecutionPolicy&& policy,
              const bool fire_reset,
              const Action* player_a_buffer,
              const Action* player_b_buffer,
              bool* doneBuffer);

    template<typename ExecutionPolicy>
    void step_render(ExecutionPolicy&& policy,
                     const bool fire_reset,
                     const Action* player_a_buffer,
                     const Action* player_b_buffer,
                     bool* doneBuffer,
                     uint8_t* frameBuffer);

    template<typename ExecutionPolicy>
    void get_data(ExecutionPolicy&& policy,
                  const bool episodic_life,
                  bool* doneBuffer,
                  float* rewardsBuffer,
                  int32_t* livesBuffer);

    template<typename ExecutionPolicy>
    void preprocess(ExecutionPolicy&& policy,
                    const uint32_t* tiaBuffer,
                    uint8_t* frameBuffer);

    template<typename ExecutionPolicy>
    void generate_frames(ExecutionPolicy&& policy,
                         const bool rescale,
                         const size_t num_channels,
                         uint8_t* imageBuffer);

    rom cart;

    size_t num_envs;
    size_t noop_reset_steps;

    State_t* states_ptr;
    uint8_t* ram_ptr;

    frame_state* frame_states_ptr;
    uint32_t* tia_update_ptr;
    uint8_t* frame_ptr;

    uint32_t* rand_states_ptr;
    uint32_t* cache_index_ptr;

    Action* minimal_actions_ptr;

    State_t* cached_states_ptr;
    uint8_t* cached_ram_ptr;
    uint8_t* cached_frame_ptr;

    frame_state* cached_frame_states_ptr;
};

template<template<typename> class Allocator>
class wrapped_environment : public wrapper
{
private:

    using super_t = wrapper;
    using State_t = typename super_t::State_t;

public:

    template<typename U>
    using VecType = agency::vector<U,Allocator<U>>;

    wrapped_environment(const rom& cart,
                        const size_t num_envs,
                        const size_t noop_reset_steps = 30);

protected:

    VecType<State_t> states_buffer;
    VecType<frame_state> frame_states_buffer;
    VecType<uint8_t> ram_buffer;
    VecType<uint32_t> tia_update_buffer;
    VecType<uint8_t> frame_buffer;

    VecType<uint32_t> rand_states_buffer;
    VecType<uint32_t> cache_index_buffer;

    VecType<Action> minimal_actions_buffer;

    VecType<State_t> cached_states_buffer;
    VecType<uint8_t> cached_ram_buffer;

    VecType<frame_state> cached_frame_states_buffer;
    VecType<uint8_t> cached_frame_buffer;
};

} // end namespace atari
} // end namespace cule
