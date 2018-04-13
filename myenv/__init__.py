from gym.envs.registration import register

register(
    id ='Firefly-v0',
    entry_point ='myenv.firefly:FireflyEnv',
)

register(
    id ='Firefly-v1',
    entry_point ='myenv.firefly2:FireflyEnv',
)
