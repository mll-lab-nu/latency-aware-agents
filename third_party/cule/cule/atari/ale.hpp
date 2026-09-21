#pragma once

#include <cule/config.hpp>

#include <cule/atari/games.hpp>

#define ALE_CASE(ID_NAME, NS_NAME, FUNC) \
case GAME_##ID_NAME:                            \
{                                               \
    FUNC(NS_NAME)                               \
    break;                                      \
}

#define GAME_SWITCH(FUNC)                                      \
switch(game_id)                                                \
{                                                              \
    ALE_CASE(DEMON_ATTACK, demonattack, FUNC)                  \
    ALE_CASE(ASTERIX, asterix, FUNC)                            \
    ALE_CASE(ATLANTIS, atlantis, FUNC)                          \
}

#define SET_TERMINAL(NAME) NAME::setTerminal(s);
#define IS_MINIMAL(NAME) value = NAME::isMinimal(a);
#define GET_REWARD(NAME) value = NAME::reward(s);
#define GET_SCORE(NAME) value = NAME::score(s);
#define GET_LIVES(NAME) value = NAME::lives(s);

namespace cule
{
namespace atari
{

struct ale
{

template<typename State_t>
static
CULE_ANNOTATION
void set_id(State_t& s, const games::GAME_TYPE& game_id)
{
    UPDATE_FIELD(s.riotData, FIELD_RIOT_GAME, game_id);
}

template<typename State_t>
static
CULE_ANNOTATION
games::GAME_TYPE get_id(State_t&s)
{
    return games::GAME_TYPE(SELECT_FIELD(s.riotData, FIELD_RIOT_GAME));
}

template<typename State_t>
static
CULE_ANNOTATION
void setTerminal(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    GAME_SWITCH(SET_TERMINAL)
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getRewards(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_REWARD)

    return value;
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getScore(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_SCORE)

    return value;
}

template<typename State_t>
static
CULE_ANNOTATION
int32_t getLives(State_t& s)
{
    using namespace games;

    GAME_TYPE game_id = get_id(s);

    int32_t value = 0;
    GAME_SWITCH(GET_LIVES)

    return value;
}

// is end of game
template<typename State_t>
static
CULE_ANNOTATION
bool isTerminal(State_t& s)
{
    return s.tiaFlags[FLAG_ALE_TERMINAL];
}

template<typename State_t>
static
CULE_ANNOTATION
bool isStarted(State_t& s)
{
    return s.tiaFlags[FLAG_ALE_STARTED];
}

template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.tiaFlags.clear(FLAG_ALE_STARTED);

    s.score = 0;
}

/** Drops illegal actions, such as the fire button in skiing. Note that this is different
  *   from the minimal set of actions. */
template<typename State_t>
static
CULE_ANNOTATION
void noopIllegalActions(State_t& s)
{
    s.sysFlags.clear(FLAG_CON_RESET);
}

}; // end namespace ale

CULE_ANNOTATION
bool isMinimal(const games::GAME_TYPE& game_id, const Action& a)
{
    using namespace games;

    bool value = false;
    GAME_SWITCH(IS_MINIMAL)

    return value;
}

// Returns the vector of the minimal set of actions needed to play
// the game.
std::vector<Action> getMinimalActionSet(const games::GAME_TYPE& game_id)
{
    std::vector<Action> actions;
    for (int i = 0; i < _ACTION_MAX; i++)
    {
        if(isMinimal(game_id, allActions[i]))
        {
            actions.push_back(allActions[i]);
        }
    }
    return actions;
}

} // end namespace atari
} // end namespace cule
