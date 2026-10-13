#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/environment.hpp>
#include <cule/atari/functors.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/m6502.hpp>
#include <cule/atari/paddles.hpp>
#include <cule/atari/palettes.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{
namespace dispatch
{

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset(ExecutionPolicy&& policy,
      Wrapper& wrap,
      uint32_t* seedBuffer)
{
    agency::bulk_invoke(policy(1),
                        initialize_functor<Environment>{},
                        &wrap.cart,
                        wrap.cached_ram_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_frame_states_ptr);

    // ALE's system RNG is separate from the episode/action RNG. Reproduce its
    // fixed hardware seed and power-on RAM. RIOT construction consumes one draw.
    std::mt19937 hardware_rng(4753849);
    for(size_t i = 0; i < wrap.cart.ram_size(); ++i)
    {
        wrap.cached_ram_ptr[i] = static_cast<uint8_t>(hardware_rng());
    }
    hardware_rng.discard(1);

    const size_t frame_size = 300 * SCREEN_WIDTH;
    const size_t boot_frames = Environment::getBaseFrames(wrap.cached_states_ptr[0])
        + Environment::getStartNumber(wrap.cached_states_ptr[0]);

    // ALE detects the display format, resets the console, initializes loadROM,
    // then runs reset_game. RAM survives each system reset.
    const size_t init_frames[] = {ENV_NOOP_FRAMES, 0, boot_frames, boot_frames + 1};
    for(const size_t frame_count : init_frames)
    {
        wrap.cached_states_ptr[0].rand = hardware_rng();
        Environment::reset(wrap.cached_states_ptr[0]);
        preprocess::reset(wrap.cached_frame_states_ptr[0]);
        for(size_t i = 0; i < frame_count; ++i)
        {
            agency::bulk_invoke(policy(1),
                                step_functor<Environment>{},
                                false,
                                wrap.cached_states_ptr,
                                wrap.tia_update_ptr,
                                nullptr,
                                nullptr,
                                nullptr);
            agency::bulk_invoke(policy(1),
                                preprocess_functor{},
                                wrap.tia_update_ptr,
                                wrap.cached_frame_states_ptr,
                                wrap.cached_frame_ptr);
        }
    }
    wrap.cached_states_ptr[0].score = Environment::ALE_t::getScore(wrap.cached_states_ptr[0]);

    for (size_t i = 1; i < wrap.noop_reset_steps; i++)
    {
        wrap.cached_states_ptr[i] = wrap.cached_states_ptr[i - 1];
        wrap.cached_frame_states_ptr[i] = wrap.cached_frame_states_ptr[i - 1];
        std::copy_n(wrap.cached_frame_ptr + (i - 1) * frame_size,
                    frame_size,
                    wrap.cached_frame_ptr + i * frame_size);

        wrap.cached_states_ptr[i].ram = (uint32_t*) (wrap.cached_ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[i - 1].ram,
                  wrap.cached_states_ptr[i - 1].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.cached_states_ptr[i].ram);

        agency::bulk_invoke(policy(1),
                            step_functor<Environment>{},
                            false,
                            wrap.cached_states_ptr + i,
                            wrap.tia_update_ptr,
                            nullptr,
                            nullptr,
                            nullptr);
        agency::bulk_invoke(policy(1),
                            preprocess_functor{},
                            wrap.tia_update_ptr,
                            wrap.cached_frame_states_ptr + i,
                            wrap.cached_frame_ptr + i * frame_size);
        wrap.cached_states_ptr[i].score = Environment::ALE_t::getScore(wrap.cached_states_ptr[i]);
    }

    for (size_t i = 0; i < wrap.size(); i++)
    {
        prng gen(wrap.rand_states_ptr[i]);
        gen.initialize(seedBuffer[i]);
        const size_t index = gen.sample() % wrap.noop_reset_steps;
        wrap.states_ptr[i] = wrap.cached_states_ptr[index];
        wrap.states_ptr[i].ram = (uint32_t *) (wrap.ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[index].ram,
                  wrap.cached_states_ptr[index].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.states_ptr[i].ram);
        wrap.frame_states_ptr[i] = wrap.cached_frame_states_ptr[index];
        wrap.cache_index_ptr[i] = index;
        std::copy_n(wrap.cached_frame_ptr + index * frame_size,
                    frame_size,
                    wrap.frame_ptr + i * frame_size);
    }
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
step(ExecutionPolicy&& policy,
     Wrapper& wrap,
     const bool fire_reset,
     const Action* playerABuffer,
     const Action* playerBBuffer,
     bool* doneBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        step_functor<Environment>{},
                        fire_reset,
                        wrap.states_ptr,
                        wrap.tia_update_ptr,
                        (Action*) playerABuffer,
                        (Action*) playerBBuffer,
                        doneBuffer);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
get_data(ExecutionPolicy&& policy,
         Wrapper& wrap,
         const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        get_data_functor<Environment>{},
                        episodic_life,
                        wrap.states_ptr,
                        doneBuffer,
                        rewardsBuffer,
                        livesBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
preprocess(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const uint32_t* tiaBuffer,
           uint8_t* frameBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        preprocess_functor{},
                        tiaBuffer,
                        wrap.frame_states_ptr,
                        frameBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
generate_frames(ExecutionPolicy&& policy,
                Wrapper& wrap,
                const bool rescale,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        generate_frame_functor{},
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        wrap.frame_ptr,
                        imageBuffer);
}

} // end namespace dispatch
} // end namespace atari
} // end namespace cule

#ifdef __CUDACC__
#include <cule/atari/cuda/dispatch.hpp>
#endif
