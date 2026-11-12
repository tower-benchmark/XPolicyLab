"""CPU contract checks; real checkpoint parity is gated by TOWER."""
import numpy as np
import pytest

from XPolicyLab.policy.OpenWAM_TOWER import model
from XPolicyLab.policy.OpenWAM_TOWER.adapter import decode_actions, raw_state


def observation(index=0):
    return dict(
        coordinate_frame='encoder_arm_base', env_idx=index, instruction=f'trial {index}',
        vision={name: {'color': np.full((12, 16, 3), [191, 19, 7], dtype=np.uint8)} for name in model.CAMERAS},
        state=dict(joint_encoder=np.zeros((2, 7)), eef_encoder=np.tile(np.eye(4), (2, 1, 1)), gripper_width_m=[.02, .04]),
    )


@pytest.fixture
def policy(monkeypatch, tmp_path):
    class Fake:
        mode = 'eef_delta'
        identity = {'seed': 42, 'package_digest': 'fixture'}
        def __init__(self, *args, **kwargs):
            self.seen = []
        def predict(self, obs):
            self.seen.append(obs)
            native = np.zeros((32, 22))
            native[:, [3, 7, 13, 17]] = 1
            native[:, [9, 19]] = .5
            result = decode_actions(self.mode, native, obs, np.array([.1, .1]))
            return dict(result, identity=self.identity)
    monkeypatch.setattr(model, 'V3OpenWAMPolicy', Fake)
    monkeypatch.setattr(model, 'get_robot_action_dim_info', lambda _: {'arm_dim': [7, 7], 'ee_dim': [1, 1]})
    return model.Model(dict(env_cfg_type='nero_dual', seed=42, package_path=str(tmp_path)))


def test_independent_anchors_batch_order_and_reset(policy):
    first, second = observation(7), observation(2)
    second['state']['eef_encoder'][0, 0, 3] = .217456789123
    policy.update_obs_batch([first, second])
    second['state']['eef_encoder'][0, 0, 3] = 9
    chunks = policy.get_action_batch([2, 7])
    assert [len(chunk) for chunk in chunks] == [16, 16]
    assert chunks[0][0]['left_ee_transform'][0, 3] == .217456789123
    assert chunks[1][0]['left_ee_transform'][0, 3] == 0
    np.testing.assert_array_equal(policy.policy.seen[0]['images'][model.CAMERAS[0]][0, 0], [191, 19, 7])
    assert np.asarray(policy.get_prediction(2)['native_actions']).shape == (32, 22)
    policy.reset()
    with pytest.raises(ValueError, match='Missing observation'):
        policy.get_action_batch([2])


def test_reject_inference_seed_and_physical_frame(policy):
    with pytest.raises(ValueError, match='fixed at 42'):
        policy.prepare_case({'inference_seed': 3})
    policy.prepare_case({'episode_seed': 3})
    obs = observation()
    obs['coordinate_frame'] = 'physical'
    with pytest.raises(ValueError, match='compensated'):
        policy.update_obs(obs)


def test_duplicate_indices(policy):
    with pytest.raises(ValueError, match='Duplicate'):
        policy.update_obs_batch([observation(), observation()])


def test_absolute_joint_encoders_and_width_projection():
    obs = dict(joint_encoder=np.arange(14).reshape(2, 7) / 10, gripper_width_m=[.02, .04])
    state = raw_state('joint_absolute', obs, np.array([.1, .1]))
    np.testing.assert_allclose(state[[1, 9]], obs['joint_encoder'][:, 1])
    native = np.tile(state, (32, 1)).astype(np.float64)
    native[:, [7, 15]] = [-1, 2]
    decoded = decode_actions('joint_absolute', native, obs, np.array([.1, .1]))
    np.testing.assert_array_equal(decoded['joint_encoder_targets'][0], obs['joint_encoder'].astype(np.float32).astype(float))
    np.testing.assert_array_equal(decoded['gripper_width_m'][0], [0, .1])
    assert decoded['j8_actuator_mask'] == [False, False]
