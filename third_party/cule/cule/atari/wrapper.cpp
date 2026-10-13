#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/dispatch.hpp>

#define ROM_CASE(NAME, FUNC, ...)                                      \
case NAME:                                                             \
{                                                                      \
    FUNC< environment<NAME> >(__VA_ARGS__);                            \
    break;                                                             \
}

#define ROM_SWITCH(FUNC, ...)                                          \
switch(cart.type())                                                    \
{                                                                      \
    ROM_CASE(ROM_4K, FUNC, __VA_ARGS__)                                \
    ROM_CASE(ROM_F8, FUNC, __VA_ARGS__)                                \
    default:                                                           \
        CULE_ASSERT(false, "Invalid rom type " << cart.type_name()); \
}

namespace cule
{
namespace atari
{

wrapper::
wrapper(const rom& cart,
        const size_t num_envs,
        const size_t noop_reset_steps)
:   cart(cart),
    num_envs(num_envs),
    noop_reset_steps(noop_reset_steps),
    states_ptr(nullptr),
    frame_states_ptr(nullptr),
    ram_ptr(nullptr),
    tia_update_ptr(nullptr),
    frame_ptr(nullptr),
    rand_states_ptr(nullptr),
    minimal_actions_ptr(nullptr),
    cached_states_ptr(nullptr),
    cached_ram_ptr(nullptr),
    cached_frame_states_ptr(nullptr),
    cached_frame_ptr(nullptr),
    cache_index_ptr(nullptr)
{}

void
wrapper::
initialize_ptrs(State_t* states_ptr,
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
                uint32_t* cache_index_ptr)
{
    this->states_ptr = states_ptr;
    this->frame_states_ptr = frame_states_ptr;
    this->ram_ptr = ram_ptr;
    this->tia_update_ptr = tia_update_ptr;
    this->frame_ptr = frame_ptr;
    this->minimal_actions_ptr = minimal_actions_ptr;
    this->rand_states_ptr = rand_states_ptr;
    this->cached_states_ptr = cached_states_ptr;
    this->cached_ram_ptr = cached_ram_ptr;
    this->cached_frame_states_ptr = cached_frame_states_ptr;
    this->cached_frame_ptr = cached_frame_ptr;
    this->cache_index_ptr = cache_index_ptr;
}

template<typename ExecutionPolicy>
void
wrapper::
reset(ExecutionPolicy&& policy,
      uint32_t* seedBuffer)
{
    ROM_SWITCH(dispatch::reset, policy, *this, seedBuffer)
}

template<typename ExecutionPolicy>
void
wrapper::
step(ExecutionPolicy&& policy,
     const bool fire_reset,
     const Action* playerABuffer,
     const Action* playerBBuffer,
     bool* doneBuffer)
{
    ROM_SWITCH(dispatch::step, policy, *this, fire_reset, playerABuffer, playerBBuffer, doneBuffer)
}

template<typename ExecutionPolicy>
void
wrapper::
step_render(ExecutionPolicy&& policy,
            const bool fire_reset,
            const Action* playerABuffer,
            const Action* playerBBuffer,
            bool* doneBuffer,
            uint8_t* frameBuffer)
{
    ROM_SWITCH(dispatch::step_render, policy, *this, fire_reset, playerABuffer, playerBBuffer, doneBuffer, frameBuffer)
}

template<typename ExecutionPolicy>
void
wrapper::
get_data(ExecutionPolicy&& policy,
         const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    ROM_SWITCH(dispatch::get_data, policy, *this, episodic_life, doneBuffer, rewardsBuffer, livesBuffer)
}

template<typename ExecutionPolicy>
void
wrapper::
preprocess(ExecutionPolicy&& policy,
           const uint32_t* tiaBuffer,
           uint8_t* frameBuffer)
{
    dispatch::preprocess(policy, *this, tiaBuffer, frameBuffer);
}

template<typename ExecutionPolicy>
void
wrapper::
generate_frames(ExecutionPolicy&& policy,
                const bool rescale,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    dispatch::generate_frames(policy, *this, rescale, num_channels, imageBuffer);
}

size_t
wrapper::
image_buffer_size(const size_t num_channels, const bool rescale) const
{
    return num_envs * num_channels * (rescale ? size_t(SCALED_SCREEN_SIZE) : cart.screen_size());
}

size_t
wrapper::
size() const
{
    return num_envs;
}

template<template<typename> class Allocator>
wrapped_environment<Allocator>::
wrapped_environment(const rom& cart,
                    const size_t num_envs,
                    const size_t noop_reset_steps)
: super_t(cart, num_envs, noop_reset_steps),
  states_buffer(num_envs, State_t{}),
  frame_states_buffer(num_envs, frame_state{}),
  ram_buffer(cart.ram_size() * num_envs, 0),
  tia_update_buffer(ENV_UPDATE_SIZE * num_envs, 0),
  frame_buffer(300 * SCREEN_WIDTH * num_envs, 0),
  rand_states_buffer(num_envs, 0),
  cached_states_buffer(noop_reset_steps, State_t{}),
  cached_ram_buffer(cart.ram_size() * noop_reset_steps, 0),
  cached_frame_states_buffer(noop_reset_steps, frame_state{}),
  cached_frame_buffer(300 * SCREEN_WIDTH * noop_reset_steps, 0),
  cache_index_buffer(num_envs, 0)
{
    auto actions = cart.minimal_actions();
    minimal_actions_buffer.assign(actions.begin(), actions.end());

    super_t::initialize_ptrs(states_buffer.data(),
                             frame_states_buffer.data(),
                             ram_buffer.data(),
                             tia_update_buffer.data(),
                             frame_buffer.data(),
                             minimal_actions_buffer.data(),
                             rand_states_buffer.data(),
                             cached_states_buffer.data(),
                             cached_ram_buffer.data(),
                             cached_frame_states_buffer.data(),
                             cached_frame_buffer.data(),
                             cache_index_buffer.data());
}

} // end namespace atari
} // end namespace cule
