import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import csv
from collections import deque

from robot_soccer_python.simulation2D import simulation2D
from robot_soccer_python.agents import Player, Pose
from robot_soccer_python.simulation import Environment, draw
from robot_soccer_python.constants import SCREEN_WIDTH, SCREEN_HEIGHT, PIX2M

DRIBBLE_ANGLE_THRESHOLD = math.pi / 2
RUN_SPEED = 1.5
DRIBBLE_SPEED = 1.0
SHOOT_SPEED = 4.0

STATE_SIZE = 24
ACTION_SIZE = 8
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 16
MEMORY_SIZE = 1000
TARGET_UPDATE = 50
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

ALPHA = 0.5

reward_history_red1 = []
reward_history_red2 = []
reward_history_yellow1 = []
reward_history_yellow2 = []

control_log_path = "ball_control_log.csv"
control_logs = []
with open(control_log_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Step", "PlayerIndex", "X", "Y"])

def compute_angle_diff(current, target):
    diff = target - current
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

def clamp_angle(angle, threshold):
    if angle > threshold:
        return threshold
    if angle < -threshold:
        return -threshold
    return angle

def resolve_collisions(players, ball, min_player_dist=0.4, min_ball_dist=0.15):

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pi = players[i]
            pj = players[j]
            dx = pi.pose.position.x - pj.pose.position.x
            dy = pi.pose.position.y - pj.pose.position.y
            dist = math.hypot(dx, dy)
            if dist < min_player_dist and dist > 1e-5:
                overlap = min_player_dist - dist
                shift_x = (dx / dist) * (overlap / 2)
                shift_y = (dy / dist) * (overlap / 2)
                pi.pose.position.x += shift_x
                pi.pose.position.y += shift_y
                pj.pose.position.x -= shift_x
                pj.pose.position.y -= shift_y

    for p in players:
        dx = ball.pose.position.x - p.pose.position.x
        dy = ball.pose.position.y - p.pose.position.y
        dist = math.hypot(dx, dy)
        player_radius = p.radius
        ball_radius = min_ball_dist
        contact_dist = player_radius + ball_radius

        if dist < contact_dist and dist > 1e-5 and ball.linear_speed < 0.05:
            overlap = min_ball_dist - dist
            shift_x = (dx / dist) * overlap
            shift_y = (dy / dist) * overlap
            ball.pose.position.x += shift_x
            ball.pose.position.y += shift_y

def maintain_ball_control(player, ball, target_speed=1.0, max_angle=math.pi / 6):
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    angle_to_ball = math.atan2(dy, dx)
    angle_error = abs(compute_angle_diff(player.pose.rotation, angle_to_ball))
    dist = math.hypot(dx, dy)

    contact_radius = player.radius + 0.15
    if dist < contact_radius and angle_error < max_angle:
        ball.linear_speed = target_speed
        ball.pose.rotation = player.pose.rotation

        offset = player.radius
        ball.pose.position.x = player.pose.position.x + offset * math.cos(player.pose.rotation)
        ball.pose.position.y = player.pose.position.y + offset * math.sin(player.pose.rotation)
        return True
    return False


def is_controlling_ball(player, ball, max_dist=0.6, max_angle=math.pi / 4, max_speed=1.5):
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)
    angle_error = abs(compute_angle_diff(player.pose.rotation, angle_to_ball))
    return dist < max_dist and angle_error < max_angle and ball.linear_speed < max_speed


def enforce_position_boundary(players, margin=0.2):
    field_width = SCREEN_WIDTH * PIX2M
    field_height = SCREEN_HEIGHT * PIX2M

    for player in players:
        r = player.radius
        player.pose.position.x = min(max(player.pose.position.x, r + margin), field_width - r - margin)
        player.pose.position.y = min(max(player.pose.position.y, r + margin), field_height - r - margin)


def red_rule_command(player, ball, opponent1, opponent2, simulation):
    field_width = SCREEN_WIDTH * PIX2M
    players = simulation.player

    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)

    dists = [(i, math.hypot(p.pose.position.x - ball.pose.position.x,
                            p.pose.position.y - ball.pose.position.y)) for i, p in enumerate(players)]
    dists.sort(key=lambda x: x[1])
    player_index = [i for i, p in enumerate(players) if p is player][0]
    rank = [i for i, _ in dists].index(player_index)

    angle_offset = (rank - 1) * (math.pi / 18)
    adjusted_angle = angle_to_ball + angle_offset

    target_x = field_width - 0.3
    target_y = (SCREEN_HEIGHT * PIX2M) / 2
    angle_to_goal = math.atan2(target_y - player.pose.position.y, target_x - player.pose.position.x)
    contact_radius = player.radius + 0.15

    if maintain_ball_control(player, ball):
        dist_to_goal = abs(player.pose.position.x - target_x)
        angle_goal_diff = abs(compute_angle_diff(player.pose.rotation, angle_to_goal))
        if angle_goal_diff < math.pi / 4 and dist_to_goal < 5.0:
            ball.linear_speed = 5.0
            ball.pose.rotation = angle_to_goal
            return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_goal))
        return (DRIBBLE_SPEED, 0.0)

    elif 0.5 < dist <= 1.0:
        raw_diff = compute_angle_diff(player.pose.rotation, angle_to_goal)
        dribble_angle = clamp_angle(raw_diff, DRIBBLE_ANGLE_THRESHOLD)
        return (DRIBBLE_SPEED, dribble_angle)

    else:
        return (RUN_SPEED, compute_angle_diff(player.pose.rotation, adjusted_angle))


def yellow_rule_command(player, ball, opponent1, opponent2, simulation):
    field_width = SCREEN_WIDTH * PIX2M
    players = simulation.player

    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)

    dists = [(i, math.hypot(p.pose.position.x - ball.pose.position.x,
                            p.pose.position.y - ball.pose.position.y)) for i, p in enumerate(players)]
    dists.sort(key=lambda x: x[1])
    player_index = [i for i, p in enumerate(players) if p is player][0]
    rank = [i for i, _ in dists].index(player_index)

    angle_offset = (rank - 1) * (math.pi / 18)
    adjusted_angle = angle_to_ball + angle_offset

    target_x = 0.3
    target_y = (SCREEN_HEIGHT * PIX2M) / 2
    angle_to_goal = math.atan2(target_y - player.pose.position.y, target_x - player.pose.position.x)
    contact_radius = player.radius + 0.15

    if maintain_ball_control(player, ball):
        dist_to_goal = abs(player.pose.position.x - target_x)
        angle_goal_diff = abs(compute_angle_diff(player.pose.rotation, angle_to_goal))
        if angle_goal_diff < math.pi / 4 and dist_to_goal < 5.0:
            ball.linear_speed = 5.0
            ball.pose.rotation = angle_to_goal
            return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_goal))
        return (DRIBBLE_SPEED, 0.0)

    elif 0.5 < dist <= 1.0:
        raw_diff = compute_angle_diff(player.pose.rotation, angle_to_goal)
        dribble_angle = clamp_angle(raw_diff, DRIBBLE_ANGLE_THRESHOLD)
        return (DRIBBLE_SPEED, dribble_angle)

    else:
        return (RUN_SPEED, compute_angle_diff(player.pose.rotation, adjusted_angle))


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, double_dqn=False):
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def get_rl_command(action, player, ball, team, simulation):
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    angle_to_ball = math.atan2(dy, dx)

    field_center_y = (SCREEN_HEIGHT * PIX2M) / 2
    if team == "red":
        goal_x = SCREEN_WIDTH * PIX2M - 0.3
    else:
        goal_x = 0.3
    goal_y = field_center_y
    angle_to_goal = math.atan2(goal_y - player.pose.position.y, goal_x - player.pose.position.x)

    if action == 0:
        # run_to_ball
        return (RUN_SPEED, compute_angle_diff(player.pose.rotation, angle_to_ball))
    elif action == 1:
        # run_left_diagonal
        return (RUN_SPEED, compute_angle_diff(player.pose.rotation, angle_to_ball + math.pi / 6))
    elif action == 2:
        # run_right_diagonal
        return (RUN_SPEED, compute_angle_diff(player.pose.rotation, angle_to_ball - math.pi / 6))
    elif action == 3:
        # dribble_to_ball
        dribble_angle = clamp_angle(compute_angle_diff(player.pose.rotation, angle_to_ball), DRIBBLE_ANGLE_THRESHOLD)
        return (DRIBBLE_SPEED, dribble_angle)
    elif action == 4:
        # dribble_left
        dribble_angle = clamp_angle(compute_angle_diff(player.pose.rotation, angle_to_ball + math.pi / 6), DRIBBLE_ANGLE_THRESHOLD)
        return (DRIBBLE_SPEED, dribble_angle)
    elif action == 5:
        # shoot_goal
        if is_controlling_ball(player, ball):
            dist_to_goal = abs(player.pose.position.x - goal_x)
            angle_diff_goal = abs(compute_angle_diff(player.pose.rotation, angle_to_goal))
            if angle_diff_goal < math.pi / 1 and dist_to_goal < 5.0:
                simulation.ball.linear_speed = 5.0
                simulation.ball.pose.rotation = angle_to_goal
        return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_goal))

    elif action == 6:
        # pass_teammate
        player_index = [i for i, p in enumerate(simulation.player) if p is player][0]
        teammates = [simulation.player[i] for i in range(len(simulation.player))
                     if i != player_index and i // 2 == player_index // 2]
        teammate = min(teammates, key=lambda p: player.pose.dist_square(p.pose))
        angle_to_teammate = math.atan2(teammate.pose.position.y - player.pose.position.y,
                                       teammate.pose.position.x - player.pose.position.x)
        if is_controlling_ball(player, ball):
            simulation.ball.linear_speed = 5.0
            simulation.ball.pose.rotation = angle_to_teammate
        return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_teammate))

    elif action == 7:
        # idle
        return (0.0, 0.0)
    else:
        return (0.0, 0.0)


def get_hybrid_command(player, ball, opponent1, opponent2, team, rl_action, simulation):
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)
    angle_diff_to_ball = compute_angle_diff(player.pose.rotation, angle_to_ball)

    if ball.linear_speed < 0.05:
        if is_controlling_ball(player, ball):
            goal_x = SCREEN_WIDTH * PIX2M - 1.0 if team == "red" else 1.0
            goal_y = (SCREEN_HEIGHT * PIX2M) / 2
            angle_to_goal = math.atan2(goal_y - player.pose.position.y, goal_x - player.pose.position.x)
            if abs(compute_angle_diff(player.pose.rotation, angle_to_goal)) < math.pi / 4:
                ball.linear_speed = 5.0
                ball.pose.rotation = angle_to_goal
                return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_goal))

        elif dist < 3.0:
            return (DRIBBLE_SPEED, clamp_angle(angle_diff_to_ball, DRIBBLE_ANGLE_THRESHOLD))
        else:
            return (RUN_SPEED, clamp_angle(angle_diff_to_ball, math.pi / 3))

    if is_controlling_ball(player, ball):
        maintain_ball_control(player, ball)
        field_w = SCREEN_WIDTH * PIX2M
        goal_x = field_w - 0.3 if team == "red" else 0.3
        goal_y = (SCREEN_HEIGHT * PIX2M) / 2
        angle_to_goal = math.atan2(goal_y - player.pose.position.y, goal_x - player.pose.position.x)
        angle_diff = abs(compute_angle_diff(player.pose.rotation, angle_to_goal))
        dist_to_goal = abs(player.pose.position.x - goal_x)

        if angle_diff < math.pi / 1 and dist_to_goal < 5.0:
            ball.linear_speed = 5.0
            ball.pose.rotation = angle_to_goal
            return (SHOOT_SPEED, compute_angle_diff(player.pose.rotation, angle_to_goal))

        if team == "red":
            return red_rule_command(player, ball, opponent1, opponent2, simulation)
        else:
            return yellow_rule_command(player, ball, opponent1, opponent2, simulation)

    rule_cmd = red_rule_command(player, ball, opponent1, opponent2, simulation) if team == "red" \
        else yellow_rule_command(player, ball, opponent1, opponent2, simulation)
    rl_cmd = get_rl_command(rl_action, player, ball, team, simulation)

    blended_lin = ALPHA * rule_cmd[0] + (1 - ALPHA) * rl_cmd[0]
    blended_ang = ALPHA * rule_cmd[1] + (1 - ALPHA) * rl_cmd[1]
    return (blended_lin, blended_ang)


def boundary_check(player, command):
    margin = 1.0
    x = player.pose.position.x
    y = player.pose.position.y
    field_width = SCREEN_WIDTH * PIX2M
    field_height = SCREEN_HEIGHT * PIX2M
    lin, ang = command
    if x < player.radius + margin:
        ang = 0
    elif x > field_width - player.radius - margin:
        ang = math.pi
    if y < player.radius + margin:
        ang = math.pi / 2
    elif y > field_height - player.radius - margin:
        ang = -math.pi / 2
    return (lin, ang)


def normalize_angle(angle):
    return math.sin(angle), math.cos(angle)

def get_state(simulation, player_index):
    player = simulation.player[player_index]
    ball = simulation.ball

    teammates = [p for i, p in enumerate(simulation.player)
                 if i != player_index and i // 2 == player_index // 2]
    opponents = [p for i, p in enumerate(simulation.player)
                 if i // 2 != player_index // 2]

    teammate = teammates[0]
    opponent1, opponent2 = opponents

    field_w = SCREEN_WIDTH * PIX2M
    field_h = SCREEN_HEIGHT * PIX2M
    field_diag = math.hypot(field_w, field_h)
    max_ball_speed = 5.0

    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist_to_ball = math.hypot(dx, dy) / field_diag
    angle_to_ball = math.atan2(dy, dx) - player.pose.rotation
    sin_atb, cos_atb = normalize_angle(angle_to_ball)

    sin_rot, cos_rot = normalize_angle(player.pose.rotation)

    return [
        player.pose.position.x / field_w,
        player.pose.position.y / field_h,
        sin_rot, cos_rot,

        ball.pose.position.x / field_w,
        ball.pose.position.y / field_h,
        *normalize_angle(ball.pose.rotation),
        ball.linear_speed / max_ball_speed,

        dist_to_ball, sin_atb, cos_atb,

        teammate.pose.position.x / field_w,
        teammate.pose.position.y / field_h,
        *normalize_angle(teammate.pose.rotation),

        opponent1.pose.position.x / field_w,
        opponent1.pose.position.y / field_h,
        *normalize_angle(opponent1.pose.rotation),

        opponent2.pose.position.x / field_w,
        opponent2.pose.position.y / field_h,
        *normalize_angle(opponent2.pose.rotation)
    ]


def compute_reward(simulation, prev_score_red, prev_score_yellow, player_index):
    reward = -0.2
    team = "red" if player_index < 2 else "yellow"
    opponent_team = "yellow" if team=="red" else "red"
    if simulation.right_goal > prev_score_red:
        reward += 1.0 if team == "red" else -1.0
    if simulation.left_goal > prev_score_yellow:
        reward += 1.0 if team == "yellow" else -1.0
    player = simulation.player[player_index]
    ball   = simulation.ball
    dist_self = math.hypot(
        ball.pose.position.x - player.pose.position.x,
        ball.pose.position.y - player.pose.position.y
    )
    teammates = [p for i,p in enumerate(simulation.player)
                 if (i//2 == player_index//2) and i!=player_index]
    opponents = [p for i,p in enumerate(simulation.player)
                 if i//2 != player_index//2]
    nearest_team  = min(teammates,  key=lambda p: p.pose.dist_square(ball.pose))
    nearest_opp   = min(opponents,  key=lambda p: p.pose.dist_square(ball.pose))
    dist_team     = math.hypot(
        ball.pose.position.x - nearest_team.pose.position.x,
        ball.pose.position.y - nearest_team.pose.position.y
    )
    dist_opp      = math.hypot(
        ball.pose.position.x - nearest_opp.pose.position.x,
        ball.pose.position.y - nearest_opp.pose.position.y
    )

    if is_controlling_ball(player, ball):
        reward += 0.5
    if dist_opp < dist_self and dist_self < 1.0:
        reward += 0.5

    field_w = SCREEN_WIDTH * PIX2M
    goal_x = field_w - 0.3 if team == "red" else 0.3
    goal_y = (SCREEN_HEIGHT * PIX2M) / 2
    angle_to_goal = math.atan2(goal_y - player.pose.position.y,
                                   goal_x - player.pose.position.x)
    angle_diff = abs(compute_angle_diff(player.pose.rotation, angle_to_goal))
    if dist_self < 0.7 and player.pose.dist_square(ball.pose) < 0.25 and dist_self < 3.0:
        if angle_diff < math.pi / 4:
            reward += 0.3

    if team=="red":
        own_half = ball.pose.position.x < field_w/2
        defend_pos = 0.5
    else:
        own_half = ball.pose.position.x > field_w/2
        defend_pos = field_w - 0.5
    if own_half:
        dist_goal = abs(player.pose.position.x - defend_pos)
        if dist_goal < 2.0 and dist_self > 1.5:
            reward += 0.05

    if ball.linear_speed > 0.5 and dist_team < 0.7:
        reward += 0.3
    if dist_team > 1.5:
        reward -= 0.1

    if ball.linear_speed < 0.05 and dist_self < 1.5:
        reward += 0.2

    angle_to_ball = math.atan2(ball.pose.position.y - player.pose.position.y,
                                ball.pose.position.x - player.pose.position.x)
    angle_error = abs(compute_angle_diff(player.pose.rotation, angle_to_ball))
    if dist_self < 0.5 and angle_error > math.pi / 2:
        reward -= 0.1

    if dist_self < 0.4 and ball.linear_speed < 0.1 and angle_error < math.pi / 6:
        reward += 0.1

    reward -= 0.01 * abs(player.angular_speed)

    return reward

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Hybrid Control: DQN vs Double DQN Soccer")
    environment = Environment(window)

    player_red1 = Player(Pose(3, 3, 0), 2, 2, 0.2)
    player_red2 = Player(Pose(3, 4, 0), 2, 2, 0.2)
    player_yellow1 = Player(Pose(6, 3, 0), 2, 2, 0.2)
    player_yellow2 = Player(Pose(6, 4, 0), 2, 2, 0.2)

    simulation = simulation2D([player_red1, player_red2, player_yellow1, player_yellow2],
                              shockable=True, full_vision=False)
    agents = [
        DQNAgent(STATE_SIZE, ACTION_SIZE, double_dqn=False),
        DQNAgent(STATE_SIZE, ACTION_SIZE, double_dqn=False),
        DQNAgent(STATE_SIZE, ACTION_SIZE, double_dqn=True),
        DQNAgent(STATE_SIZE, ACTION_SIZE, double_dqn=True)
    ]

    num_episodes = 100
    max_steps = 200
    total_steps = 0
    update_target_every = TARGET_UPDATE
    control_counter = [0, 0, 0, 0]

    prev_score_red = simulation.right_goal
    prev_score_yellow = simulation.left_goal

    clock = pygame.time.Clock()

    for episode in range(num_episodes):
        done = False
        states = [get_state(simulation, i) for i in range(4)]
        episode_rewards = [0.0] * 4

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            actions = [agents[i].select_action(states[i]) for i in range(4)]

            commands = []
            for i, player in enumerate(simulation.player):
                team = "red" if i < 2 else "yellow"

                if is_controlling_ball(player, simulation.ball):
                    control_counter[i] += 1
                else:
                    control_counter[i] = 0

                if control_counter[i] > 3:
                    teammates = [simulation.player[j] for j in range(4) if j != i and j // 2 == i // 2]
                    nearest_teammate = min(teammates, key=lambda p: player.pose.dist_square(p.pose))
                    dist_to_teammate = math.hypot(
                        player.pose.position.x - nearest_teammate.pose.position.x,
                        player.pose.position.y - nearest_teammate.pose.position.y
                    )

                    if dist_to_teammate > 2:
                        goal_x = SCREEN_WIDTH * PIX2M - 0.3 if team == "red" else 0.3
                        goal_y = (SCREEN_HEIGHT * PIX2M) / 2
                        angle_to_goal = math.atan2(goal_y - player.pose.position.y,
                                                   goal_x - player.pose.position.x)
                        simulation.ball.linear_speed = 5.0
                        simulation.ball.pose.rotation = angle_to_goal
                    else:
                        angle_to_teammate = math.atan2(
                            nearest_teammate.pose.position.y - player.pose.position.y,
                            nearest_teammate.pose.position.x - player.pose.position.x
                        )
                        simulation.ball.linear_speed = 5.0
                        simulation.ball.pose.rotation = angle_to_teammate

                    control_counter[i] = 0

                if team == "red":
                    opponent1 = simulation.player[2]
                    opponent2 = simulation.player[3]
                else:
                    opponent1 = simulation.player[0]
                    opponent2 = simulation.player[1]
                cmd = get_hybrid_command(player, simulation.ball, opponent1, opponent2, team, actions[i], simulation)
                cmd = boundary_check(player, cmd)
                commands.append(cmd)

            simulation.set_commands(commands)
            simulation.update()
            resolve_collisions(simulation.player, simulation.ball)
            enforce_position_boundary(simulation.player)

            controlled = False
            for i, p in enumerate(simulation.player):
                if is_controlling_ball(p, simulation.ball):
                    control_logs.append([
                        episode, step, i,
                        round(p.pose.position.x, 3),
                        round(p.pose.position.y, 3)
                    ])
                    controlled = True
                    break

            if not controlled:
                control_logs.append([episode, step, -1, -1, -1])

            if step % 10 == 0:
                with open(control_log_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(control_logs)
                control_logs.clear()

            simulation.ball.linear_speed *= 0.95
            if simulation.ball.linear_speed < 0.01:
                simulation.ball.linear_speed = 0.0

            draw(simulation, window, environment)
            pygame.display.update()

            next_states = [get_state(simulation, i) for i in range(4)]
            rewards = [compute_reward(simulation, prev_score_red, prev_score_yellow, i) for i in range(4)]

            for i in range(4):
                done_flag = simulation.right_goal > prev_score_red or simulation.left_goal > prev_score_yellow
                agents[i].memory.push(states[i], actions[i], rewards[i], next_states[i], done_flag)
                agents[i].update()
                episode_rewards[i] += rewards[i]

            states = next_states
            total_steps += 1

            if simulation.right_goal > prev_score_red or simulation.left_goal > prev_score_yellow:
                prev_score_red = simulation.right_goal
                prev_score_yellow = simulation.left_goal
                done = True
                break

            if total_steps % update_target_every == 0:
                for agent in agents:
                    agent.update_target()

            clock.tick(60)

        print(f"Episode {episode + 1}: "
              f"Rewards = {[f'{r:.2f}' for r in episode_rewards]}")

        reward_history_red1.append(episode_rewards[0])
        reward_history_red2.append(episode_rewards[1])
        reward_history_yellow1.append(episode_rewards[2])
        reward_history_yellow2.append(episode_rewards[3])

        with open(control_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(control_logs)
        control_logs.clear()

    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"agent_{i}.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(moving_average(reward_history_red1), label="Red1 (DQN)", linestyle='--')
    plt.plot(moving_average(reward_history_red2), label="Red2 (DQN)", linestyle='--')
    plt.title("DQN Agents (Red Team)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("DQN_rewards_with_MA.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(moving_average(reward_history_yellow1), label="Yellow1 (DoubleDQN)", linestyle='--')
    plt.plot(moving_average(reward_history_yellow2), label="Yellow2 (DoubleDQN)", linestyle='--')
    plt.title("Double DQN Agents (Yellow Team)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Double_DQN_rewards_with_MA.png")
    plt.show()

    pygame.quit()

if __name__ == "__main__":
    main()
