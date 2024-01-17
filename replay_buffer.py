import random


class ExperienceReplayBuffer:
    def __init__(self, buffer_size):
        self.cnt = 0
        self.buffer_size = buffer_size
        self.s_t = []
        self.a_t = []
        self.r = []
        self.s_t_n = []
        self.terminal = []

    def add_experience(
            self,
            node_node,  # adjacent mat
            action_num,
            reward,
            node_node_n,
            terminal
    ):
        if self.cnt == self.buffer_size:
            self.s_t.pop(0)
            self.a_t.pop(0)
            self.r.pop(0)
            self.s_t_n.pop(0)
            self.terminal.pop(0)

        self.s_t.append(node_node)
        self.a_t.append(action_num)
        self.r.append(reward)
        self.s_t_n.append(node_node_n)
        self.terminal.append(terminal)
        self.cnt += 1

    def from_state_sequence(
            self,
            states,
            actions,
            rewards,
            n_step,
            n_experience
    ):
        cnt = 0
        for i in range(n_experience):
            if i + n_step > len(rewards) - 1:
                break
            self.add_experience(
                states[i],
                actions[i],
                rewards[i + n_step],
                states[i + n_step],
                False if i + n_step == len(states) - 1 else True
            )
            cnt += 1
        return n_experience - cnt

    def sample(self, n_sample):
        indexes = random.sample(range(self.cnt), n_sample)
        result = ([], [], [], [], [])
        for i in indexes:
            result[0].append(self.s_t[i])
            result[1].append(self.a_t[i])
            result[2].append(self.r[i])
            result[3].append(self.s_t_n[i])
            result[4].append(self.terminal[i])
        return result
