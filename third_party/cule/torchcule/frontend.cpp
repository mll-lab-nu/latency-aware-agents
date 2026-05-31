#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <string>

#include <torchcule/atari_env.hpp>

namespace py = pybind11;

using AtariAction = cule::atari::Action;
using AtariRom = cule::atari::rom;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<AtariAction>(m, "AtariAction", py::arithmetic())
    .value("NOOP", AtariAction::ACTION_NOOP)
    .value("RIGHT", AtariAction::ACTION_RIGHT)
    .value("LEFT", AtariAction::ACTION_LEFT)
    .value("DOWN", AtariAction::ACTION_DOWN)
    .value("UP", AtariAction::ACTION_UP)
    .value("FIRE", AtariAction::ACTION_FIRE)
    .value("RESET", AtariAction::ACTION_RESET)
    .value("UPRIGHT", AtariAction::ACTION_UPRIGHT)
    .value("UPLEFT", AtariAction::ACTION_UPLEFT)
    .value("DOWNRIGHT", AtariAction::ACTION_DOWNRIGHT)
    .value("DOWNLEFT", AtariAction::ACTION_DOWNLEFT)
    .value("UPFIRE", AtariAction::ACTION_UPFIRE)
    .value("RIGHTFIRE", AtariAction::ACTION_RIGHTFIRE)
    .value("LEFTFIRE", AtariAction::ACTION_LEFTFIRE)
    .value("DOWNFIRE", AtariAction::ACTION_DOWNFIRE)
    .value("UPRIGHTFIRE", AtariAction::ACTION_UPRIGHTFIRE)
    .value("UPLEFTFIRE", AtariAction::ACTION_UPLEFTFIRE)
    .value("DOWNRIGHTFIRE", AtariAction::ACTION_DOWNRIGHTFIRE)
    .value("DOWNLEFTFIRE", AtariAction::ACTION_DOWNLEFTFIRE)
    .export_values()
    ;

    py::class_<AtariRom>(m, "AtariRom")
    .def(py::init<const std::string&>())
    .def("minimal_actions", &AtariRom::minimal_actions)
    .def("ram_size", &AtariRom::ram_size)
    .def("screen_width", &AtariRom::screen_width)
    ;

    py::class_<AtariEnv>(m, "AtariEnv")
    .def(py::init<const AtariRom&, const size_t, const size_t>())
    .def("initialize", [](AtariEnv &env,
                          uint64_t statesBuffer,
                          uint64_t frameStatesBuffer,
                          uint64_t ramBuffer,
                          uint64_t tiaUpdateBuffer,
                          uint64_t frameBuffer,
                          uint64_t romIndicesBuffer,
                          uint64_t minimalActionsBuffer,
                          uint64_t randStatesBuffer,
                          uint64_t cachedStatesBuffer,
                          uint64_t cachedRamBuffer,
                          uint64_t cachedFrameStatesBuffer,
                          uint64_t cachedTiaUpdateBuffer,
                          uint64_t cacheIndexBuffer)
        {
            env.initialize_ptrs(reinterpret_cast<cule::atari::state*>(statesBuffer),
                                reinterpret_cast<cule::atari::frame_state*>(frameStatesBuffer),
                                reinterpret_cast<uint8_t*>(ramBuffer),
                                reinterpret_cast<uint32_t*>(tiaUpdateBuffer),
                                reinterpret_cast<uint8_t*>(frameBuffer),
                                reinterpret_cast<uint32_t*>(romIndicesBuffer),
                                reinterpret_cast<AtariAction*>(minimalActionsBuffer),
                                reinterpret_cast<uint32_t*>(randStatesBuffer),
                                reinterpret_cast<cule::atari::state*>(cachedStatesBuffer),
                                reinterpret_cast<uint8_t*>(cachedRamBuffer),
                                reinterpret_cast<cule::atari::frame_state*>(cachedFrameStatesBuffer),
                                reinterpret_cast<uint32_t*>(cachedTiaUpdateBuffer),
                                reinterpret_cast<uint32_t*>(cacheIndexBuffer));
        }
    )
    .def("reset", [](AtariEnv &env, uint64_t seedBuffer)
        {
            env.reset((uint32_t*) seedBuffer);
        }
    )
    .def("reset_states", [](AtariEnv &env)
        {
            env.reset_states();
        }
    )
    .def("step", [](AtariEnv &env, const bool fire_reset, uint64_t playerABuffer, uint64_t playerBBuffer, uint64_t doneBuffer)
        {
            env.step(fire_reset,
                     reinterpret_cast<AtariAction*>(playerABuffer),
                     reinterpret_cast<AtariAction*>(playerBBuffer),
                     reinterpret_cast<bool*>(doneBuffer));
        }
    )
    .def("get_data", [](AtariEnv& env, const bool episodic_life, uint64_t doneBuffer, uint64_t rewardBuffer, uint64_t livesBuffer)
        {
            env.get_data(episodic_life,
                         reinterpret_cast<bool*>(doneBuffer),
                         reinterpret_cast<float*>(rewardBuffer),
                         reinterpret_cast<int32_t*>(livesBuffer));
        }
    )
    .def("generate_frames", [](AtariEnv& env, const bool rescale, const bool last_frame, const size_t num_channels, uint64_t imageBuffer)
        {
            env.generate_frames(rescale,
                                last_frame,
                                num_channels,
                                reinterpret_cast<uint8_t*>(imageBuffer));
        }
    )
    .def("step_decision", [](AtariEnv& env,
                             const bool episodic_life,
                             const size_t obs_stride_raw_frames,
                             const size_t num_channels,
                             const int64_t hold_duration,
                             const int64_t ordering_policy,
                             const int64_t max_episode_steps,
                             const int64_t fixed_latency_raw_frames,
                             const int64_t latency_distribution_type,
                             const double latency_distribution_mean,
                             const double latency_distribution_std,
                             const double latency_distribution_min,
                             const double latency_distribution_max,
                             const int64_t batch_size,
                             const int64_t queue_capacity,
                             uint64_t actions,
                             uint64_t active,
                             uint64_t env_step_idx,
                             uint64_t decision_step_idx,
                             uint64_t next_action_id,
                             uint64_t current_latency_raw_frames,
                             uint64_t current_applied_action,
                             uint64_t persistence_remaining,
                             uint64_t pending_actions,
                             uint64_t pending_ready_raw_frames,
                             uint64_t pending_action_ids,
                             uint64_t pending_head,
                             uint64_t pending_count,
                             uint64_t scratch_actions,
                             uint64_t scratch_ready_raw_frames,
                             uint64_t scratch_action_ids,
                             uint64_t decision_queued_actions,
                             uint64_t decision_latency_raw_frames,
                             uint64_t decision_ready_raw_frames,
                             uint64_t decision_timeout,
                             uint64_t packed_decision,
                             uint64_t decision_rewards,
                             uint64_t decision_rewards_clipped,
                             uint64_t decision_terminated,
                             uint64_t decision_truncated,
                             uint64_t decision_raw_frames_advanced,
                             uint64_t episode_step_idx,
                             uint64_t latency_rng_state,
                             uint64_t action_set,
                             uint64_t raw_action_values,
                             uint64_t still_active,
                             uint64_t doneBuffer,
                             uint64_t step_rewards,
                             uint64_t livesBuffer,
                             uint64_t scoresBuffer,
                             uint64_t imageBuffer)
        {
            env.step_decision(
                episodic_life,
                obs_stride_raw_frames,
                num_channels,
                hold_duration,
                ordering_policy,
                max_episode_steps,
                fixed_latency_raw_frames,
                latency_distribution_type,
                latency_distribution_mean,
                latency_distribution_std,
                latency_distribution_min,
                latency_distribution_max,
                batch_size,
                queue_capacity,
                reinterpret_cast<int64_t*>(actions),
                reinterpret_cast<bool*>(active),
                reinterpret_cast<int64_t*>(env_step_idx),
                reinterpret_cast<int64_t*>(decision_step_idx),
                reinterpret_cast<int64_t*>(next_action_id),
                reinterpret_cast<int64_t*>(current_latency_raw_frames),
                reinterpret_cast<int64_t*>(current_applied_action),
                reinterpret_cast<int64_t*>(persistence_remaining),
                reinterpret_cast<int64_t*>(pending_actions),
                reinterpret_cast<int64_t*>(pending_ready_raw_frames),
                reinterpret_cast<int64_t*>(pending_action_ids),
                reinterpret_cast<int64_t*>(pending_head),
                reinterpret_cast<int64_t*>(pending_count),
                reinterpret_cast<int64_t*>(scratch_actions),
                reinterpret_cast<int64_t*>(scratch_ready_raw_frames),
                reinterpret_cast<int64_t*>(scratch_action_ids),
                reinterpret_cast<int64_t*>(decision_queued_actions),
                reinterpret_cast<int64_t*>(decision_latency_raw_frames),
                reinterpret_cast<int64_t*>(decision_ready_raw_frames),
                reinterpret_cast<bool*>(decision_timeout),
                reinterpret_cast<double*>(packed_decision),
                reinterpret_cast<float*>(decision_rewards),
                reinterpret_cast<float*>(decision_rewards_clipped),
                reinterpret_cast<bool*>(decision_terminated),
                reinterpret_cast<bool*>(decision_truncated),
                reinterpret_cast<int64_t*>(decision_raw_frames_advanced),
                reinterpret_cast<int64_t*>(episode_step_idx),
                reinterpret_cast<int64_t*>(latency_rng_state),
                reinterpret_cast<AtariAction*>(action_set),
                reinterpret_cast<AtariAction*>(raw_action_values),
                reinterpret_cast<bool*>(still_active),
                reinterpret_cast<bool*>(doneBuffer),
                reinterpret_cast<float*>(step_rewards),
                reinterpret_cast<int32_t*>(livesBuffer),
                reinterpret_cast<int32_t*>(scoresBuffer),
                reinterpret_cast<uint8_t*>(imageBuffer));
        }
    )
    .def("set_cuda", &AtariEnv::set_cuda)
    .def("state_size", &AtariEnv::state_size)
    .def("frame_state_size", &AtariEnv::frame_state_size)
    .def("tia_update_size", &AtariEnv::tia_update_size)
    ;
}
