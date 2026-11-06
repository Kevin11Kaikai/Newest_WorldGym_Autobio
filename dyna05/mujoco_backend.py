"""AutoBio physical packet execution; imported only when a real run is requested.

The original Stage 4 bridge supplies the exact policy-state encoder. No guessed
state ordering or replacement checkpoint is used here.
"""
from __future__ import annotations
import copy
import numpy as np
from .core import InitialState, Observation


class MuJoCoTube:
    evidence_kind = 'mujoco'

    def __init__(self, task, observe):
        import mujoco
        self.mj, self.task, self.observe = mujoco, task, observe
        self.bounds = task.model.actuator_ctrlrange[:7].copy()
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.geom_ids = [mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                         for name in ('1/aloha:leftfinger', '1/cap_cyl')]
        if min(self.geom_ids) < 0:
            raise ValueError('Pickup geometry names differ from the audited AutoBio task')
        if not np.isclose(task.model.opt.timestep * 100, .2, rtol=0, atol=1e-10):
            raise ValueError('Expected 10 controls x 10 physics steps = 0.2 seconds')

    def _observation(self):
        height = float(self.task.object.get_body_pose(self.task.data).pos[2])
        contact = any({int(c.geom1), int(c.geom2)} == set(self.geom_ids)
                      for c in self.task.data.contact[:self.task.data.ncon])
        official = bool(self.task.check())
        if official != (height > .925 and contact):
            raise ValueError('Official task success differs from the frozen Stage 4 rule')
        return Observation(np.asarray(self.observe(self.task), dtype=float).copy(), height,
                           float(contact), official, False)

    def _snapshot(self):
        state = np.empty(self.mj.mj_stateSize(self.task.model, self.state_spec))
        self.mj.mj_getState(self.task.model, self.task.data, state, self.state_spec)
        return {'integration': state}

    def initial(self, seed):
        self.task.reset(int(seed))
        self.mj.mj_forward(self.task.model, self.task.data)
        return InitialState(seed, self._snapshot(), self._observation())

    def reset(self, initial):
        # Reset task-specific Python state before restoring ALL integration state.
        self.task.reset(initial.seed)
        self.mj.mj_setState(self.task.model, self.task.data,
                           initial.snapshot['integration'].copy(), self.state_spec)
        self.mj.mj_forward(self.task.model, self.task.data)
        restored = self._snapshot()['integration']
        if not np.allclose(restored, initial.snapshot['integration'], atol=1e-9, rtol=0):
            raise ValueError('MuJoCo integration state restore is not exact within tolerance')
        return self._observation()

    def step(self, packet):
        packet = np.asarray(packet)
        if packet.shape != (10, 7) or not np.isfinite(packet).all():
            raise ValueError('Expected finite 10 x 7 actuator commands')
        if (packet < self.bounds[:, 0]).any() or (packet > self.bounds[:, 1]).any():
            raise ValueError('Out-of-bound actuator command')
        start_time = float(self.task.data.time)
        for command in packet:
            self.task.data.ctrl[:7] = command
            for _ in range(10):
                self.task.step_and_log({})
                if self.task.data.warning.number.any():
                    raise RuntimeError('MuJoCo numerical warning; run is invalid')
        if not np.isclose(self.task.data.time - start_time, .2, atol=1e-9, rtol=0):
            raise ValueError('Packet duration drift')
        return self._observation()

    def close(self):
        self.task.finish()
