#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/m6502.hpp>
#include <cule/atari/paddles.hpp>
#include <cule/atari/palettes.hpp>
#include <cule/atari/prng.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{

template<typename Environment_t>
struct initialize_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const rom* cart,
                    uint8_t* ram_buffer,
                    State_t* states_buffer,
                    frame_state* frame_states_buffer) const
    {
        using Accessor_t = typename Environment_t::Accessor_t;
        using ALE_t = typename Environment_t::ALE_t;
        using Controller_t = typename Environment_t::Controller_t;

        const bool use_paddles = cart->use_paddles();
        const bool swap_paddles = cart->swap_paddles() || cart->swap_ports();
        const bool left_difficulty_B = cart->player_left_difficulty_B();
        const bool right_difficulty_B = cart->player_right_difficulty_B();
        const bool is_ntsc = cart->is_ntsc();
        const bool hmove_blanks = cart->allow_hmove_blanks();
        const games::GAME_TYPE game_id = cart->game_id();
        const uint8_t * rom_buffer = cart->data();

        State_t& s = states_buffer[self.index()];

        s.ram = (uint32_t*)(ram_buffer + (cart->ram_size() * self.index()));
        s.rom = rom_buffer;
        s.tia_update_buffer = nullptr;

        Accessor_t::initialize(s);

        // update controller settings
        Controller_t::set_flags(s, use_paddles, swap_paddles, left_difficulty_B, right_difficulty_B);

        // set the game id
        ALE_t::set_id(s, game_id);

        s.tiaFlags.template change<FLAG_TIA_IS_NTSC>(is_ntsc);
        s.tiaFlags.template change<FLAG_TIA_HMOVE_ALLOW>(hmove_blanks);
        s.tiaFlags.template change<FLAG_TIA_Y_SHIFT>(true);

        s.tiaFlags.set(FLAG_ALE_TERMINAL);
        s.tiaFlags.set(FLAG_ALE_STARTED);
        s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);

        const int32_t num_actions = game_id == games::GAME_ASTERIX ? 1 : 0;
        const Action starting_action = ACTION_FIRE;

        Environment_t::setStartNumber(s, num_actions);
        Environment_t::setStartAction(s, starting_action);

        frame_state& fs = frame_states_buffer[self.index()];
        preprocess::reset(fs);
        fs.tiaFlags.template change<FLAG_TIA_IS_NTSC>(is_ntsc);
        fs.tiaFlags.template change<FLAG_TIA_HMOVE_ALLOW>(hmove_blanks);
        fs.tiaFlags.template change<FLAG_TIA_Y_SHIFT>(true);
    }
};

template<typename Environment_t>
struct step_functor
{
    template<class Agent, class State_t>
    void operator()(Agent& self,
                    const bool fire_reset,
                    State_t* states_buffer,
                    uint32_t* tia_update_buffer,
                    Action* player_a_buffer,
                    Action* player_b_buffer,
                    bool* done_buffer) const
    {
        if((done_buffer != nullptr) && done_buffer[self.index()])
        {
            return;
        }

        State_t& s = states_buffer[self.index()];

        s.tia_update_buffer = nullptr;
        if(tia_update_buffer != nullptr)
        {
            s.tia_update_buffer = tia_update_buffer + (ENV_UPDATE_SIZE * self.index());
        }

        Action player_a_action = ACTION_NOOP;
        Action player_b_action = ACTION_NOOP;

        if(player_a_buffer != nullptr)
        {
            player_a_action = player_a_buffer[self.index()];
        }
        if(player_b_buffer != nullptr)
        {
            player_b_action = player_b_buffer[self.index()];
        }

        if(fire_reset && s.tiaFlags[FLAG_ALE_LOST_LIFE])
        {
            player_a_action = ACTION_FIRE;
            s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);
        }

        Environment_t::act(s, player_a_action, player_b_action);
    }
};

template<typename Environment_t>
struct get_data_functor
{
    template<class Agent, typename State_t>
    void operator()(Agent& self,
                    const bool episodic_life,
                    State_t* states_buffer,
                    bool* done_buffer,
                    float* rewards_buffer,
                    int32_t* lives_buffer)
    {
        using ALE_t = typename Environment_t::ALE_t;

        if((done_buffer != nullptr) && done_buffer[self.index()])
        {
            return;
        }

        State_t& s = states_buffer[self.index()];

        const uint32_t old_lives = lives_buffer[self.index()];
        const uint32_t new_lives = ALE_t::getLives(s);
        lives_buffer[self.index()] = new_lives;

        const bool lost_life = new_lives < old_lives;
        s.tiaFlags.template change<FLAG_ALE_LOST_LIFE>(lost_life);

        rewards_buffer[self.index()] += ALE_t::getRewards(s);
        done_buffer[self.index()] |= s.tiaFlags[FLAG_ALE_TERMINAL] || (episodic_life && lost_life);

        s.score = ALE_t::getScore(s);
    }
};

struct preprocess_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const uint32_t* tia_update_buffer,
                    frame_state* frame_states_buffer,
                    uint8_t* frame_buffer)
    {
        frame_state& fs = frame_states_buffer[self.index()];
        fs.srcBuffer = tia_update_buffer + (ENV_UPDATE_SIZE * self.index());

        fs.framePointer = &frame_buffer[self.index() * 300 * SCREEN_WIDTH];

        preprocess::state_to_buffer(fs);
    }
};

struct generate_frame_functor
{
    template<class Agent>
    void operator()(Agent& self,
                    const size_t num_channels,
                    const size_t screen_height,
                    const bool rescale,
                    const uint8_t* frame_buffer,
                    uint8_t* image_buffer)
    {
        const uint8_t* framePointer = &frame_buffer[self.index() * 300 * SCREEN_WIDTH];

        if(rescale)
        {
            uint8_t * image_buffer_temp = &image_buffer[self.index() * num_channels * SCALED_SCREEN_SIZE];
            apply_rescale(screen_height, num_channels, image_buffer_temp, framePointer);
        }
        else
        {
            uint8_t * image_buffer_temp = &image_buffer[self.index() * num_channels * screen_height * SCREEN_WIDTH];
            apply_palette(screen_height, num_channels, image_buffer_temp, framePointer);
        }
    }
};

} // end namespace atari
} // end namespace cule
