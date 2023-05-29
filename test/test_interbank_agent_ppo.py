import unittest,interbank_agent_ppo


class PPOTestCase(unittest.TestCase):

    def test_environment(self):
        from stable_baselines3.common.env_checker import check_env
        env = interbank_agent_ppo.InterbankPPO()
        check_env(env)
        self.assertIsInstance(env, gymnasium.Env)


if __name__ == '__main__':
    unittest.main()
