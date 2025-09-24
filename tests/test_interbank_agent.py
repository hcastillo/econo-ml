import unittest
import interbank_agent
import gymnasium


class PPOTestCase(unittest.TestCase):
    """
    It tests the environment of the ReinforcementLearning algorithm
    and also the correct type of class
    """

    def test_environment(self):
        # from stable_baselines3.common.env_checker import check_env
        env = interbank_agent.InterbankAgent()
        # check_env(env)
        self.assertIsInstance(env, gymnasium.Env)


if __name__ == '__main__':
    unittest.main()
