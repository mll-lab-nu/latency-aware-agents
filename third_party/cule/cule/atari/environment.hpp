#pragma once

#include <cule/config.hpp>

#include <cule/atari/accessors.hpp>
#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/m6532.hpp>
#include <cule/atari/tia.hpp>

#include <agency/agency.hpp>

#include <iomanip>
#include <random>

namespace cule
{
namespace atari
{

template<typename, typename> struct interrupt;
template<typename, typename, typename> struct mmc;
template<typename, typename, typename> struct m6502;
template<int32_t> struct rom_accessor;
template<typename> struct stack;

template<int32_t ROM_FORMAT>
struct environment
{

using Accessor_t = rom_accessor<ROM_FORMAT>;
using ALE_t = ale;
using Controller_t = controller;

using M6532_t = m6532<Controller_t>;
using TIA_t = tia<M6532_t, Controller_t>;

using MMC_t = mmc<Accessor_t, M6532_t, TIA_t>;
using Stack_t = stack<MMC_t>;
using Interrupt_t = interrupt<MMC_t, Stack_t>;
using M6502_t = m6502<MMC_t, Stack_t, Interrupt_t>;

template<typename State_t>
static
CULE_ANNOTATION
void increment(State_t& s)
{
    INC_FIELD(s.frameData, FIELD_FRAME_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
void setFrameNumber(State_t& s, const int frame_number)
{
    UPDATE_FIELD(s.frameData, FIELD_FRAME_NUMBER, frame_number);
}

template<typename State_t>
static
CULE_ANNOTATION
int getFrameNumber(State_t& s)
{
    return SELECT_FIELD(s.frameData, FIELD_FRAME_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
void setStartNumber(State_t& s, int num_actions)
{
    UPDATE_FIELD(s.frameData, FIELD_START_NUMBER, num_actions);
}

template<typename State_t>
static
CULE_ANNOTATION
int getStartNumber(State_t& s)
{
    return SELECT_FIELD(s.frameData, FIELD_START_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
void setStartAction(State_t& s, const Action& starting_action)
{
    UPDATE_FIELD(s.frameData, FIELD_START_ACTION, starting_action);
}

template<typename State_t>
static
CULE_ANNOTATION
Action getStartAction(State_t& s)
{
    return Action(SELECT_FIELD(s.frameData, FIELD_START_ACTION));
}

// ALE performs two four-frame soft resets. Demon Attack and Atlantis also
// reset when applying their default game mode.
template<typename State_t>
static
CULE_ANNOTATION
int getBaseFrames(State_t& s)
{
    const auto game = ALE_t::get_id(s);
    const int mode_reset = game == games::GAME_DEMON_ATTACK || game == games::GAME_ATLANTIS;
    return ENV_NOOP_FRAMES + (2 + mode_reset) * ENV_RESET_FRAMES;
}

/** Resets the system to its start state. */
template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    // Reset ALE
    ALE_t::reset(s);

    // Reset the paddles
    Controller_t::reset(s);

    // Reset timers
    M6532_t::reset(s);

    // A system reset restores the cartridge bank before reading the reset vector.
    Accessor_t::initialize(s);

    // Reset the processor
    M6502_t::reset(s);

    // Reset the TIA
    TIA_t::reset(s);

    // Reset the frame number
    setFrameNumber(s, 0);
}

/** Actually emulates the emulator for a given number of steps. */
template<typename State_t>
static
CULE_ANNOTATION
void emulate(State_t& s)
{
    if(!s.tiaFlags[FLAG_TIA_PARTIAL])
    {
        TIA_t::startFrame(s);
    }

    s.tiaFlags.set(FLAG_TIA_PARTIAL);

    // update paddle position at every step
    Controller_t::applyAction(s);
    M6502_t::run(s);

    if(!s.tiaFlags[FLAG_TIA_PARTIAL])
    {
        TIA_t::finishFrame(s);

        // update ale reward
        ALE_t::setTerminal(s);

        increment(s);
    }
}

/** This applies an action exactly one time step. Helper function to act(). */
template<typename State_t>
static
CULE_ANNOTATION
void act(State_t& s, const Action& player_a_action, const Action& player_b_action = ACTION_NOOP)
{
    if (ALE_t::isTerminal(s) && ALE_t::isStarted(s))
    {
        return;
    }

    // Convert illegal actions into NOOPs; actions such as reset are always illegal
    ALE_t::noopIllegalActions(s);

    const int32_t frame_number = getFrameNumber(s);
    const int32_t start_number = getStartNumber(s);
    const int32_t base_frames = getBaseFrames(s);

    if(frame_number < ENV_NOOP_FRAMES)
    {
        Controller_t::set_action(s, ACTION_NOOP);
    }
    else if(frame_number < base_frames)
    {
        Controller_t::set_action(s, ACTION_RESET);
    }
    else if(frame_number < (base_frames + start_number))
    {
        Controller_t::set_action(s, getStartAction(s));
    }
    else
    {
        Controller_t::set_actions(s, player_a_action, player_b_action);
    }

    s.tiaFlags.template change<FLAG_ALE_STARTED>(frame_number >= (base_frames + start_number));

    emulate(s);
}

}; // end class environment

} // end namespace atari
} // end namespace cule

