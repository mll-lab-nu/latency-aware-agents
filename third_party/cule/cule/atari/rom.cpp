#pragma once

#include <cule/config.hpp>
#include <cule/md5.hpp>

#include <cule/atari/ale.hpp>
#include <cule/atari/games.hpp>
#include <cule/atari/rom.hpp>

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <zlib.h>

namespace cule
{
namespace atari
{
namespace detail
{
ROM_FORMAT detect_type(const int32_t size)
{
    if(size == 4096)
    {
        return ROM_4K;
    }
    if(size == 8192)
    {
        return ROM_F8;
    }
    return ROM_NOT_SUPPORTED;
}
} // end namespace detail

rom::rom(const std::string& filename)
    : _ram_size(0),
      _rom_size(0)
{
    if(filename.size())
    {
        reset(filename);
    }
}

rom::rom(const rom& other)
    : _ram_size(other._ram_size),
      _rom_size(other._rom_size),
      _type(other._type),
      _gameId(other._gameId),
      _md5(other._md5),
      _filename(other._filename),
      image(other.image),
      _minimal_actions(other._minimal_actions)
{}

void rom::reset(const std::string& filename)
{
    _filename = filename;

    gzFile f = gzopen(_filename.c_str(), "rb");
    CULE_ASSERT(f != nullptr, "gzopen failed to open " << _filename);

    image.resize(MAX_ROM_SIZE);
    _rom_size = gzread(f, image.data(), MAX_ROM_SIZE);
    gzclose(f);

    compute_md5();
    set_game_id();

    _type = detail::detect_type(rom_size());

    _ram_size = 128;
}

std::vector<Action> const& rom::minimal_actions() const
{
    return _minimal_actions;
}

std::string rom::file_name() const
{
    return _filename;
}

std::string rom::game_name() const
{
    return value_or_default(games::ROM_ATTR_Name);
}

std::string rom::type_name() const
{
    return names[type()];
}

bool rom::swap_ports() const
{
    return value_or_default(games::ROM_ATTR_SwapPorts) == "YES";
}

bool rom::use_paddles() const
{
    return value_or_default(games::ROM_ATTR_ControllerLeft) == "PADDLES";
}

bool rom::swap_paddles() const
{
    return value_or_default(games::ROM_ATTR_SwapPaddles) == "YES";
}

bool rom::allow_hmove_blanks() const
{
    return value_or_default(games::ROM_ATTR_HmoveBlanks) == "YES";
}

bool rom::player_left_difficulty_B() const
{
    return value_or_default(games::ROM_ATTR_LeftDifficulty) == "B";
}

bool rom::player_right_difficulty_B() const
{
    return value_or_default(games::ROM_ATTR_RightDifficulty) == "B";
}

size_t rom::ram_size() const
{
    return _ram_size;
}

size_t rom::rom_size() const
{
    return _rom_size;
}

size_t rom::screen_height() const
{
    const bool is_pal_m = game_name().find("PAL-M") != std::string::npos;
    return (is_ntsc() || is_pal_m) ? std::stoi(value_or_default(games::ROM_ATTR_Height)) : PAL_SCREEN_HEIGHT;
}

size_t rom::screen_width() const
{
    return SCREEN_WIDTH;
}

size_t rom::screen_size() const
{
    return screen_height() * screen_width();
}

ROM_FORMAT rom::type() const
{
    return _type;
}

games::GAME_TYPE rom::game_id() const
{
    return _gameId;
}

std::string rom::md5() const
{
    return _md5;
}

uint8_t const* rom::data() const
{
    return image.data();
}

bool rom::is_supported() const
{
    return type() == ROM_4K || type() == ROM_F8;
}

bool rom::is_ntsc() const
{
    return game_name().find("PAL") == std::string::npos;
}

std::ostream& operator<< (std::ostream& os, rom const& cart)
{
    os << "Cartridge\n";
    os << "\tFile       : " << cart.file_name() << "\n";
    os << "\tName       : " << cart.game_name() << "\n";
    os << "\tController : " << (cart.use_paddles() ? "Paddles" : "Joystick") << "\n";
    os << "\tSwapped    : " << ((cart.swap_paddles() || cart.swap_ports()) ? "Yes" : "No") << "\n";
    os << "\tLeft  Diff : " << (cart.player_left_difficulty_B()  ? "B" : "A") << "\n";
    os << "\tRight Diff : " << (cart.player_right_difficulty_B() ? "B" : "A") << "\n";
    os << "\tType       : " << cart.type_name() << "\n";
    os << "\tDisplay    : " << (cart.is_ntsc() ? "NTSC" : "PAL") << "\n";
    os << "\tRAM Size   : " << cart.ram_size() << "\n";
    os << "\tROM Size   : " << cart.rom_size() << "\n";
    os << "\tMD5        : " << cart.md5() << "\n";
    return os;
}

std::string rom::value_or_default(const games::ROM_ATTR attr) const
{
    CULE_ASSERT(rom_size() != 0, "Rom has not been properly initialized");

    using namespace games;
    const std::string n = rom_attr_data.at(game_id())[attr];
    return !n.empty() ? n : default_attr[attr];
}

bool rom::has_banks() const
{
    return type() == ROM_F8;
}

void rom::set_game_id()
{
    _gameId = games::rom_game_map.at(md5());
    _minimal_actions = getMinimalActionSet(_gameId);
}

void rom::compute_md5()
{
    std::array<uint8_t,MD5_DIGEST_LENGTH> md5_result;
    MD5(image.data(), _rom_size, md5_result.data());

    std::ostringstream buffer;
    buffer.fill('0');
    buffer.setf(std::ios::hex, std::ios::basefield);
    for(auto c : md5_result)
    {
        buffer << std::setw(2) << uint32_t(c);
    }
    _md5 = buffer.str();
}

const char* rom::names[_ROM_MAX];
static void initStrings()
{
    rom::names[ROM_4K]="4K";
    rom::names[ROM_F8]="F8";
    rom::names[ROM_NOT_SUPPORTED]="NOT SUPPORTED";
}

class rom_initialize
{
  public:
    rom_initialize()
    {
        initStrings();
    }
};
rom_initialize startup;

} // end namespace atari
} // end namespace cule
