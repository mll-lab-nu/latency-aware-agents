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
namespace atlantis
{

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
bool isGameOver(State& s)
{
    return cule::atari::ram::read(s.ram, 0xF1) == 0xFF;
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return isGameOver(s) ? 0 : cule::atari::ram::read(s.ram, 0xF1);
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
    if (isGameOver(s))
    {
        return s.score;
    }
    return 100 * cule::atari::games::getDecimalScore(s, 0xA2, 0xA3, 0xA1);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace atlantis
} // end namespace games
} // end namespace atari
} // end namespace cule
