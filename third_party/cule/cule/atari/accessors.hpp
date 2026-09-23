#pragma once

#include <cule/config.hpp>

#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

template<int32_t> struct rom_accessor;

template<>
struct rom_accessor<ROM_4K>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t&, const maddr_t&, const uint8_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        return s.rom[addr & 0x0FFF];
    }
};

template<>
struct rom_accessor<ROM_F8>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t& s)
    {
        s.bank = 1;
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t&)
    {
        switch(addr & 0x0FFF)
        {
            case 0x0FF8:
            {
                s.bank = 0;
                break;
            }
            case 0x0FF9:
            {
                s.bank = 1;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        write(s, addr, 0);
        return s.rom[(s.bank * 4096) + (addr & 0x0FFF)];
    }
};

} // end namespace atari
} // end namespace cule
