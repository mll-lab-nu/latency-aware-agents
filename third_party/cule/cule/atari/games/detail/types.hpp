#pragma once

#include <cule/config.hpp>

#include <cule/atari/games/detail/attributes.hpp>

#include <array>
#include <map>
#include <string>

namespace cule
{
namespace atari
{
namespace games
{

static std::map<std::string, GAME_TYPE> rom_game_map =
{
    { "f0e0addc07971561ab80d9abe1b8d333", GAME_DEMON_ATTACK },
    { "89a68746eff7f266bbf08de2483abe55", GAME_ASTERIX },
    { "9ad36e699ef6f45d9eb6c4cf90475c9f", GAME_ATLANTIS },
};

const std::array<std::string,_ROM_ATTR_MAX> rom_attr_names =
{{
        "Cartridge.Manufacturer",
        "Cartridge.ModelNo",
        "Cartridge.Name",
        "Cartridge.Note",
        "Cartridge.Rarity",
        "Cartridge.Sound",
        "Cartridge.Type",
        "Console.LeftDifficulty",
        "Console.RightDifficulty",
        "Console.TelevisionType",
        "Console.SwapPorts",
        "Controller.Left",
        "Controller.Right",
        "Controller.SwapPaddles",
        "Display.Format",
        "Display.YStart",
        "Display.Height",
        "Display.Phosphor",
        "Display.PPBlend",
        "Emulation.HmoveBlanks"
    }
};

const std::array<std::string,_ROM_ATTR_MAX> default_attr =
{{
        "",            // Cartridge.Manufacturer
        "",            // Cartridge.ModelNo
        "Untitled",    // Cartridge.Name
        "",            // Cartridge.Note
        "",            // Cartridge.Rarity
        "MONO",        // Cartridge.Sound
        "AUTO-DETECT", // Cartridge.Type
        "B",           // Console.LeftDifficulty
        "B",           // Console.RightDifficulty
        "COLOR",       // Console.TelevisionType
        "NO",          // Console.SwapPorts
        "JOYSTICK",    // Controller.Left
        "JOYSTICK",    // Controller.Right
        "NO",          // Controller.SwapPaddles
        "AUTO-DETECT", // Display.Format
        "34",          // Display.YStart
        "210",         // Display.Height
        "NO",          // Display.Phosphor
        "77",          // Display.PPBlend
        "YES"          // Emulation.HmoveBlanks
    }
};

static std::map<GAME_TYPE, std::array<std::string,_ROM_ATTR_MAX>> rom_attr_data =
{
    { GAME_DEMON_ATTACK, {{"", "", "Demon Attack", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ASTERIX, {{"Atari", "CX2696", "Asterix (1988) (Atari) (Prototype) (NTSC)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ATLANTIS, {{"Imagic", "", "Atlantis", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
};

} // end namespace games
} // end namespace atari
} // end namespace cule
