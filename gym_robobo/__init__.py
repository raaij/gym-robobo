import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Robobo-v0',
    entry_point='gym_robobo.env:RoboboEnv',
)