@ray.remote
class VI_server_v1(object):
    def __init__(self, size):
        self.v_current = [0] * size
        self.pi = [0] * size
        self.v_new = [0] * size

    def get_value_and_policy(self):
        return self.v_current, self.pi

    def update(self, update_index, update_v, update_pi):
        self.v_new[update_index] = update_v
        self.pi[update_index] = update_pi

    def get_error_and_update(self):
        max_error = 0
        for i in range(len(self.v_current)):
            error = abs(self.v_new[i] - self.v_current[i])
            if error > max_error:
                max_error = error
            self.v_current[i] = self.v_new[i]

        return max_error


@ray.remote
def VI_worker_v1(VI_server, data, worker_id, update_state):
    env, workers_num, beta, epsilon = data
    A = env.GetActionSpace()
    S = env.GetStateSpace()

    # get shared variable
    V, _ = ray.get(VI_server.get_value_and_policy.remote())

    # bellman backup

    # INSERT YOUR CODE HERE
    tmp_v = []
    max_v = float('-inf')
    max_a = 0
    for action in range(A):
        immediate_reward = env.GetReward(update_state, action)
        future_reward = 0
        for next_state in range(S):
            future_reward += env.GetTransitionProb(update_state, action, next_state) * V[next_state]

        tmp_v.append(immediate_reward + beta * future_reward)

    max_v = np.max(tmp_v)
    max_a = np.argmax(tmp_v)

    VI_server.update.remote(update_state, max_v, max_a)
    # return ith worker
    return worker_id


def sync_value_iteration_distributed_v1(env, beta=0.999, epsilon=0.01, workers_num=4, stop_steps=2000):
    S = env.GetStateSpace()
    VI_server = VI_server_v1.remote(S)
    workers_list = []
    data_id = ray.put((env, workers_num, beta, epsilon))

    start = 0
    # start the all worker, store their id in a list
    for i in range(workers_num):
        w_id = VI_worker_v1.remote(VI_server, data_id, i, start)
        workers_list.append(w_id)
        start += 1

    error = float('inf')
    while error > epsilon:
        for update_state in range(start, S):
            # Wait for one worker finishing, get its reuslt, and delete it from list
            finished_worker_id = ray.wait(workers_list, num_returns=1, timeout=None)[0][0]
            finish_worker = ray.get(finished_worker_id)
            workers_list.remove(finished_worker_id)

            # start a new worker, and add it to the list
            w_id = VI_worker_v1.remote(VI_server, data_id, finish_worker, update_state)
            workers_list.append(w_id)
        start = 0
        error = ray.get(VI_server.get_error_and_update.remote())

    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi