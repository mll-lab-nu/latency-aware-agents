#pragma once

#include <cule/atari/flags.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/wrapper.hpp>

#include <cule/atari/games/detail/attributes.hpp>

#include <cstdint>

class AtariEnv : public cule::atari::wrapper
{
    private:
      using super_t = cule::atari::wrapper;

    public:
      AtariEnv(const cule::atari::rom& cart,
               const size_t num_envs,
               const size_t noop_reset_steps);

      ~AtariEnv();

      void reset(uint32_t* seedBuffer);

      void step(const bool fire_reset,
                const cule::atari::Action* playerABuffer,
                const cule::atari::Action* playerBBuffer,
                bool* doneBuffer);

      void get_data(const bool episodic_life,
                    bool* doneBuffer,
                    float* rewardsBuffer,
                    int32_t* livesBuffer);

      void generate_frames(const bool rescale,
                           const size_t num_channels,
                           uint8_t* imageBuffer);

      void generate_obs84(uint8_t* imageBuffer);

      void insert_other_stream(uint64_t stream);

      void insert_this_stream(uint64_t stream);

      void step_decision(const bool episodic_life,
                         const size_t obs_stride_raw_frames,
                         const size_t obs_frame_count,
                         const size_t num_channels,
                         const bool hold_action,
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
                         const int64_t* actions,
                         const bool* active,
                         const bool* admitted,
                         int64_t* env_step_idx,
                         int64_t* decision_step_idx,
                         int64_t* next_action_id,
                         int64_t* current_latency_raw_frames,
                         int64_t* current_applied_action,
                         int64_t* pending_actions,
                         int64_t* pending_ready_raw_frames,
                         int64_t* pending_action_ids,
                         int64_t* pending_head,
                         int64_t* pending_count,
                         int64_t* scratch_actions,
                         int64_t* scratch_ready_raw_frames,
                         int64_t* scratch_action_ids,
                         int64_t* decision_queued_actions,
                         int64_t* decision_latency_raw_frames,
                         int64_t* decision_ready_raw_frames,
                         bool* decision_timeout,
                         double* packed_decision,
                         float* decision_rewards,
                         float* decision_rewards_clipped,
                         bool* decision_terminated,
                         bool* decision_truncated,
                         int64_t* decision_raw_frames_advanced,
                         int64_t* episode_step_idx,
                         int64_t* latency_rng_state,
                         int64_t* profile_latency_raw_frames,
                         cule::atari::Action* action_set,
                         cule::atari::Action* raw_action_values,
                         bool* still_active,
                         bool* doneBuffer,
                         float* step_rewards,
                         int32_t* livesBuffer,
                         int32_t* scoresBuffer,
                         uint8_t* imageBuffer);

      void set_cuda(const bool use_cuda, const int32_t gpu_id);

      size_t state_size();

      size_t frame_state_size();

      size_t tia_update_size();

      template<typename ExecutionPolicy>
      ExecutionPolicy& get_policy();

    private:
      void* cule_par;
      bool use_cuda;
      int32_t gpu_id;
};
