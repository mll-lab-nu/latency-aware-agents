#pragma once

#include <cule/config.hpp>
#include <cule/atari/actions.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{
namespace games
{
namespace asterix
{

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_UP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_DOWN:
    case ACTION_UPRIGHT:
    case ACTION_UPLEFT:
    case ACTION_DOWNRIGHT:
    case ACTION_DOWNLEFT:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t rawLives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xD3) & 0xF;
}

template<typename State>
CULE_ANNOTATION
bool isGameOver(State& s)
{
    const int death_counter = cule::atari::ram::read(s.ram, 0xC7);
    return (death_counter == 0x01) && (rawLives(s) == 1);
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return isGameOver(s) ? 0 : rawLives(s);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(isGameOver(s));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xE0, 0xDF, 0xDE);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace asterix
} // end namespace games
} // end namespace atari
} // end namespace cule
