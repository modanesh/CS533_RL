import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory_remote import ReplayBuffer_remote

import matplotlib.pyplot as plt
from custom_cartpole import CartPoleEnv


# matplotlib inline

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000, temp_dir="../../tmp/")

FloatTensor = torch.FloatTensor



# # Set the Env name and action space for CartPole
# ENV_NAME = 'CartPole-v0'
# # Move left, Move right
# ACTION_DICT = {"LEFT": 0, "RIGHT": 1}
# # Register the environment
# env_CartPole = gym.make(ENV_NAME)
#
# # Set result saveing floder
# result_floder = ENV_NAME
# result_file = ENV_NAME + "/results.txt"
# if not os.path.isdir(result_floder):
#     os.mkdir(result_floder)

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)


def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.show()


hyperparams_CartPole = {'epsilon_decay_steps': 100000, 'final_epsilon': 0.1, 'batch_size': 32, 'update_steps': 10,
    'memory_size': 2000, 'beta': 0.99, 'model_replace_freq': 2000, 'learning_rate': 0.0003, 'use_target_model': True,
                        'initial_epsilon': 1}

import gym


@ray.remote
class DQN_model_server(object):
    def __init__(self, env, memory, action_space=2, test_interval=50):

        self.collector_done = False
        self.evaluator_done = False

        self.env = env
        # self.max_episode_steps = env._max_episode_steps
        self.max_episode_steps = 200

        self.beta = hyperparams_CartPole['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyperparams_CartPole['final_epsilon']
        self.epsilon_decay_steps = hyperparams_CartPole['epsilon_decay_steps']
        self.batch_size = hyperparams_CartPole['batch_size']

        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        self.previous_q_models = []
        self.results = [0] * (self.batch_size + 1)
        self.reuslt_count = 0
        self.episode = 0
        self.test_interval = test_interval
        self.memory = memory

        state = env.reset()
        input_len = len(state)
        output_len = action_space

        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyperparams_CartPole['learning_rate'])

        self.use_target_model = hyperparams_CartPole['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)

        # #         memory: Store and sample experience replay.
        #         self.memory = ReplayBuffer(hyper_params['memory_size'])

        self.batch_size = hyperparams_CartPole['batch_size']
        self.update_steps = hyperparams_CartPole['update_steps']
        self.model_replace_freq = hyperparams_CartPole['model_replace_freq']


    def get_steps(self):
        return self.steps

    def update_batch(self):

        # if len(memory) < self.batch_size or self.steps % self.update_steps != 0:
        #     return
        # print(len(self.memory.remote()))
        batch = self.memory.sample.remote(self.batch_size)
        (states, actions, reward, next_states, is_terminal) = ray.get(batch)

        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values
        _, self.q_values = self.eval_model.predict_batch(states)
        self.q_values = self.q_values[batch_index, actions]

        # Calculate target
        if self.use_target_model:
            actions, self.q_next = self.target_model.predict_batch(next_states)
            self.q_next = self.q_next[batch_index, actions]
        else:
            actions, self.q_next = self.eval_model.predict_batch(next_states)
            self.q_next = self.q_next[batch_index, actions]

        # INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        self.q_target = []

        for i in range(len(reward)):

            if terminal[i] == 1:
                self.q_target.append(reward[i])
            else:
                self.q_target.append(reward[i] + self.beta * self.q_next[i])
    
        self.q_target = FloatTensor(self.q_target)

        # update model
        self.eval_model.fit(self.q_values, self.q_target)

        if(np.random.randint(100)==4):
            print("==========",self.q_values[0],self.q_target[0])
            # print("..................................................", self.evaluate())

        # score = self.evaluate()
        # f_results = open("./results_8_4.txt", "a+")
        # f_results.write(str(score) + "\n")
        # f_results.close()

        if self.episode // self.test_interval + 1 > len(self.previous_q_models):
            model_id = ray.put(self.eval_model)
            self.previous_q_models.append(model_id)

        self.steps += 10
        return self.steps

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def replace_target(self):

        return self.target_model.replace(self.eval_model)

        # evalutor

    # def add_result(self, result):
    #     self.results[num] = result

    def get_reuslts(self):
        return self.results


    def ask_evaluation(self):
        if len(self.previous_q_models) > self.reuslt_count:
            num = self.reuslt_count
            evluation_q_model = self.previous_q_models[num]
            self.reuslt_count += 1
            return evluation_q_model, False, num
        else:
            if self.episode >= training_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

    def add_episode(self):

        self.episode += 1

    # def evaluate(self, trials=10):
    #     environment = gym.make("CartPole-v0")
    #
    #     total_reward = 0
    #     for _ in tqdm(range(trials), desc="Evaluating"):
    #         state = environment.reset()
    #         done = False
    #         steps = 0
    #
    #         while steps < self.max_episode_steps and not done:
    #             steps += 1
    #             action = self.greedy_policy(state)
    #             state, reward, done, _ = environment.step(action)
    #             total_reward += reward
    #
    #     avg_reward = total_reward / trials
        # print(avg_reward)
        # f = open(result_file, "a+")
        # f.write(str(avg_reward) + "\n")
        # f.close()
        # if avg_reward >= self.best_reward:
        #     self.best_reward = avg_reward
            # self.save_model()
        # return avg_reward


@ray.remote
def collecting_server(env, DQN_model_server, memory, action_space=2):
    update_steps = hyperparams_CartPole['update_steps']
    model_replace_freq = hyperparams_CartPole['model_replace_freq']
    use_target_model = hyperparams_CartPole['use_target_model']
    initial_epsilon = hyperparams_CartPole['initial_epsilon']
    final_epsilon = hyperparams_CartPole['final_epsilon']
    epsilon_decay_steps = hyperparams_CartPole['epsilon_decay_steps']
    model_server = DQN_model_server
    global_step = 0


    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate


    def explore_or_exploit_policy(state, global_step):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = linear_decrease(initial_epsilon, final_epsilon, global_step, epsilon_decay_steps)


        if p < epsilon:
            # return action
            return randint(0, action_space - 1)
        else:
            # return action
            return ray.get(model_server.greedy_policy.remote(state))


    # for episode in tqdm(range(test_interval), desc="Training"):
    for episode in tqdm(range(1250), desc="Training"):
        print("Collection server, episode:", episode)
        state = env.reset()
        done = False
        steps = 0

        # while steps < self.max_episode_steps and not done:  # INSERT YOUR CODE HERE  # add experience from explore-exploit policy to memory  # update the model every 'update_steps' of experience  # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
        while steps <= 200 and not done:  # INSERT YOUR CODE HERE  # add experience from explore-exploit policy to memory  # update the model every 'update_steps' of experience  # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
            global_step+=1
            action = explore_or_exploit_policy(state, global_step)
            next_state, reward, done, _ = env.step(action)

            # experience_sample=(states, actions, reward, next_states, done)
            memory.add.remote(state, action, reward, next_state, done)

            ### update model

            if (steps % update_steps == 0):
                global_step = ray.get(model_server.update_batch.remote())

            if (steps % model_replace_freq == 0 and use_target_model):
                ray.get(model_server.replace_target.remote())

            steps = steps + 1
            state = next_state

        DQN_model_server.add_episode.remote()


@ray.remote
def evaluation_server(env, DQN_model_server, trials=30):

    model_server = DQN_model_server
    while True:

        eval_model_id, done, num = ray.get(model_server.ask_evaluation.remote())
        eval_model = ray.get(eval_model_id)
        if done:
            break
        if eval_model == []:
            continue
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluation"):
            state = env.reset()
            done = False
            steps = 0
            while steps < 200 and not done:
                action = eval_model.predict(state)
                observation, reward, done, info = env.step(action)

                total_reward += reward
                steps += 1
                state = observation
        # model_server.add_result.remote(total_reward / trials)
        f_results = open("./results.txt", "a+")
        f_episodes = open("./episodes.txt", "a+")
        f_results.write(str(total_reward / trials) + "\n")
        if num == None:
            num = "None"
        f_episodes.write(str(num) + "\n")
        f_results.close()
        f_episodes.close()




class distributed_DQN_agent():

    #define the model server
    # define replay buffer
    # define env

    def __init__(self, env, collectors_num=4, evaluators_num=2, action_space=2):

        self.memory = ReplayBuffer_remote.remote(hyperparams_CartPole['memory_size'])
        self.model_server = DQN_model_server.remote(env, self.memory, action_space)

        self.cw_num = collectors_num
        self.ew_num = evaluators_num
        self.env = env  # self.max_episode_steps = env._max_episode_steps

    def learn_and_evaluate(self):
        worker_id = []
        # evaluators_id = []

        for i in range(self.cw_num):
            simulator = CartPoleEnv()
            worker_id.append(collecting_server.remote(simulator, self.model_server, self.memory, action_space=2))

        # ray.wait(collectors_id, len(collectors_id))
        #
        for j in range(self.ew_num):
            simulator = self.env
            worker_id.append(evaluation_server.remote(simulator, self.model_server))

        ray.wait(worker_id, len(worker_id))

        return ray.get(self.model_server.get_reuslts.remote())


####################

training_episodes, test_interval = 7000, 50
ENV_NAME = 'CartPole_distributed'

cartpole = CartPoleEnv()
# env_CartPole = gym.make(CartPoleEnv)
# print(cartpole.reset())

agent = distributed_DQN_agent(cartpole, collectors_num=8, evaluators_num=4)
result = agent.learn_and_evaluate()
plot_result(result, test_interval, ["batch_update with target_model"])
