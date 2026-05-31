#pragma once

#include <cule/config.hpp>

namespace cule
{
namespace atari
{
namespace games
{

enum GAME_TYPE : uint8_t
{
    GAME_DEMON_ATTACK,
};

enum ROM_ATTR
{
    ROM_ATTR_Manufacturer,
    ROM_ATTR_ModelNo,
    ROM_ATTR_Name,
    ROM_ATTR_Note,
    ROM_ATTR_Rarity,
    ROM_ATTR_Sound,
    ROM_ATTR_Type,
    ROM_ATTR_LeftDifficulty,
    ROM_ATTR_RightDifficulty,
    ROM_ATTR_TelevisionType,
    ROM_ATTR_SwapPorts,
    ROM_ATTR_ControllerLeft,
    ROM_ATTR_ControllerRight,
    ROM_ATTR_SwapPaddles,
    ROM_ATTR_Format,
    ROM_ATTR_YStart,
    ROM_ATTR_Height,
    ROM_ATTR_Phosphor,
    ROM_ATTR_PPBlend,
    ROM_ATTR_HmoveBlanks,
    _ROM_ATTR_MAX
};

} // end namespace games
} // end namespace atari
} // end namespace cule
