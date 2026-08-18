from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.gymnasium_asterix as gymnasium_asterix
import scripts.scaffold_gymnasium_asterix as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper, GymnasiumActionLatencyWrapper


ACTION_LABELS = ["noop", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft"]


class _FakeAle:
    def __init__(self) -> None:
        self.episode_frame = 0
    def getEpisodeFrameNumber(self): return self.episode_frame
    def getAvailableModes(self): return [0]
    def getAvailableDifficulties(self): return [0]


class _AsterixProbeEnv(gym.Env):
    action_space = gym.spaces.Discrete(9)
    observation_space = gym.spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
    def __init__(self) -> None:
        self.ale = _FakeAle()
        self.metadata = {"render_modes": ["rgb_array"]}
        self.spec = SimpleNamespace(id="ALE/Asterix-v5", entry_point="fake:AtariEnv", max_episode_steps=None, kwargs={"render_mode": "rgb_array"})
        self.actions: list[int] = []
        self.closed = False
    def get_action_meanings(self): return ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.ale.episode_frame = 0
        return np.zeros((210,160,3),dtype=np.uint8), {"lives":3,"episode_frame_number":0}
    def step(self, action):
        self.actions.append(action); self.ale.episode_frame += 1
        return np.full((210,160,3),self.ale.episode_frame,dtype=np.uint8),0.0,False,False,{"lives":3,"episode_frame_number":self.ale.episode_frame}
    def render(self): return np.full((210,160,3),7,dtype=np.uint8)
    def close(self): self.closed=True


def test_asterix_wrapper_exposes_chw_obs_and_native_rgb(monkeypatch):
    native=_AsterixProbeEnv(); captured:dict[str,Any]={}
    monkeypatch.setattr(gymnasium_asterix,"_register_ale",lambda:None)
    def fake_make(env_id:str,**kwargs:Any): captured.update(env_id=env_id,kwargs=kwargs); return native
    monkeypatch.setattr(gymnasium_asterix.gym,"make",fake_make)
    env=gymnasium_asterix.make_latency_sensitive_asterix_env(base_env_id="ALE/Asterix-v5",render_mode="rgb_array",screen_size=84,noop_max=0,base_make_kwargs={"frameskip":1})
    obs,_=env.reset(seed=4); next_obs,*_=env.step(8)
    assert captured=={"env_id":"ALE/Asterix-v5","kwargs":{"render_mode":"rgb_array","frameskip":1}}
    assert obs.shape==(1,84,84) and next_obs.shape==(1,84,84)
    assert native.actions==[8]
    assert env.render().shape==(210,160,3)


def test_asterix_random_noop_reset_uses_native_noop(monkeypatch):
    native=_AsterixProbeEnv()
    monkeypatch.setattr(gymnasium_asterix,"_register_ale",lambda:None)
    monkeypatch.setattr(gymnasium_asterix.gym,"make",lambda *args,**kwargs:native)
    env=gymnasium_asterix.make_latency_sensitive_asterix_env(base_env_id="ALE/Asterix-v5",render_mode="rgb_array",screen_size=84,noop_max=30,base_make_kwargs={})
    env.reset(seed=3)
    assert 1 <= len(native.actions) <= 30
    assert set(native.actions)=={0}


def test_l1_scheduler_advances_four_raw_frames(monkeypatch):
    native=_AsterixProbeEnv()
    monkeypatch.setattr(gymnasium_asterix,"_register_ale",lambda:None)
    monkeypatch.setattr(gymnasium_asterix.gym,"make",lambda *args,**kwargs:native)
    task=gymnasium_asterix.make_latency_sensitive_asterix_env(base_env_id="ALE/Asterix-v5",render_mode="rgb_array",screen_size=84,noop_max=0,base_make_kwargs={})
    env=GymnasiumActionCatalogWrapper(task,list(range(9)))
    env=GymnasiumActionLatencyWrapper(env,fixed_latency_raw_frames=1,env_fps=60,obs_fps=15,hold_policy="hold",ordering_policy="issue_order_fifo",noop_action=0)
    env.reset(seed=0); env.step(5); env.step(8)
    assert native.actions==[0,5,5,5,5,8,8,8]


def test_scaffold_reports_ntsc_and_random_noop(monkeypatch,capsys):
    native=_AsterixProbeEnv()
    monkeypatch.setattr(scaffold.gym,"make",lambda *args,**kwargs:native)
    monkeypatch.setattr(scaffold.importlib.metadata,"version",lambda package:{"gymnasium":"0.29.1","ale-py":"0.8.1"}[package])
    scaffold.main([]); output=yaml.safe_load(capsys.readouterr().out)
    assert output["probe_report"]["native_env_fps"]==60
    assert output["probe_report"]["video_standard"]=="NTSC"
    assert output["probe_report"]["action_space"]["meanings"]==["NOOP","UP","RIGHT","LEFT","DOWN","UPRIGHT","UPLEFT","DOWNRIGHT","DOWNLEFT"]
    assert output["env_config"]["env"]["make_kwargs"]["noop_max"]==30
    assert output["env_config"]["env"]["action_order"]==ACTION_LABELS


@pytest.mark.parametrize("config_path",[
    "configs/examples/gymnasium/asterix/latency_bench_eval.yaml",
    "configs/examples/gymnasium/asterix/small_model_train.yaml",
    "configs/examples/gymnasium/asterix/small_model_eval.yaml",
    "configs/examples/gymnasium/asterix/starvla_eval.yaml",
])
def test_asterix_configs_share_contract_and_l1(config_path):
    config=load_config(config_path); env_cfg=config["env"]
    clock=EnvClock(env_fps=env_cfg["env_fps"],obs_fps=env_cfg["obs_fps"])
    assert env_cfg["env_id"]==gymnasium_asterix.GYMNASIUM_ASTERIX_ENV_ID
    assert env_cfg["make_kwargs"]["noop_max"]==30
    assert env_cfg["make_kwargs"]["base_make_kwargs"]["frameskip"]==1
    assert env_cfg["make_kwargs"]["base_make_kwargs"]["repeat_action_probability"]==0.0
    assert env_cfg["action_order"]==ACTION_LABELS
    assert env_cfg["action_map"]==dict(zip(ACTION_LABELS,range(9)))
    assert clock.obs_stride_raw_frames==4
    assert clock.latency_ms_to_raw_frames(config["latency"]["fixed_latency_ms"])==1
