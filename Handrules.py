import math
import time
import pygame
from robot_soccer_python.simulation2D import simulation2D
from robot_soccer_python.agents import Player, Pose
from robot_soccer_python.simulation import Environment, draw
from robot_soccer_python.constants import SCREEN_WIDTH, SCREEN_HEIGHT, PIX2M

DRIBBLE_ANGLE_THRESHOLD = math.pi / 2
PASS_THRESHOLD = 0.8

def compute_angle_diff(current, target):
    diff = target - current
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def clamp_angle(angle, threshold):
    return max(-threshold, min(threshold, angle))

def ball_is_in_front(player, ball):
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    angle_to_ball = math.atan2(dy, dx)
    angle_diff = compute_angle_diff(player.pose.rotation, angle_to_ball)
    return abs(angle_diff) < math.pi / 4

def best_pass_teammate(player, teammates):
    px, py = player.pose.position.x, player.pose.position.y
    min_score = -float('inf')
    best = None
    for t in teammates:
        tx, ty = t.pose.position.x, t.pose.position.y
        dx, dy = tx - px, ty - py
        dist = math.hypot(dx, dy)
        angle = math.atan2(ty - py, tx - px)
        goal_x = SCREEN_WIDTH * PIX2M - 0.3 if px < SCREEN_WIDTH * PIX2M / 2 else 0.3
        goal_y = (SCREEN_HEIGHT * PIX2M) / 2
        goal_angle = math.atan2(goal_y - ty, goal_x - tx)
        alignment = math.cos(goal_angle - angle)
        score = alignment * 2 - dist
        if score > min_score:
            min_score = score
            best = t
    return best


last_pass_time = {0: 0, 2: 0}

def red_strategy(player, ball, opponent, teammate=None, teammates=[], idx=0, current_time=0):
    field_width = SCREEN_WIDTH * PIX2M
    field_mid = field_width / 2
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)
    dribble_angle = compute_angle_diff(player.pose.rotation, angle_to_ball)
    dribble_angle = clamp_angle(dribble_angle, DRIBBLE_ANGLE_THRESHOLD)

    if dist > 1.0:
        return 1.2, compute_angle_diff(player.pose.rotation, angle_to_ball)
    elif dist > 0.5:
        return 0.8, dribble_angle
    else:
        if not ball_is_in_front(player, ball):
            return 0.3, compute_angle_diff(player.pose.rotation, angle_to_ball)

        if teammates and (current_time - last_pass_time.get(idx, 0)) > 2.0:
            teammate = best_pass_teammate(player, teammates)
            if teammate:
                tx, ty = teammate.pose.position.x, teammate.pose.position.y
                angle_pass = math.atan2(ty - player.pose.position.y, tx - player.pose.position.x)
                ball.linear_speed = 5.0
                ball.pose.rotation = angle_pass
                last_pass_time[idx] = current_time

                teammate_target_x = SCREEN_WIDTH * PIX2M - 0.3 if player.pose.position.x < SCREEN_WIDTH * PIX2M / 2 else 0.3
                teammate_target_y = (SCREEN_HEIGHT * PIX2M) / 2
                teammate.pose.rotation = math.atan2(teammate_target_y - teammate.pose.position.y, teammate_target_x - teammate.pose.position.x)
                return 1.5, compute_angle_diff(player.pose.rotation, angle_pass)

        target_x = field_width - 0.3
        target_y = (SCREEN_HEIGHT * PIX2M) / 2
        angle_goal = math.atan2(target_y - player.pose.position.y, target_x - player.pose.position.x)
        ball.linear_speed = 5.0
        ball.pose.rotation = angle_goal
        return 1.5, compute_angle_diff(player.pose.rotation, angle_goal)


last_pass_time_yellow = {1: 0, 3: 0}

def yellow_strategy(player, ball, opponent, teammate=None, teammates=[], idx=1, current_time=0):
    field_width = SCREEN_WIDTH * PIX2M
    dx = ball.pose.position.x - player.pose.position.x
    dy = ball.pose.position.y - player.pose.position.y
    dist = math.hypot(dx, dy)
    angle_to_ball = math.atan2(dy, dx)
    dribble_angle = compute_angle_diff(player.pose.rotation, angle_to_ball)
    dribble_angle = clamp_angle(dribble_angle, DRIBBLE_ANGLE_THRESHOLD)

    if dist > 1.0:
        return 1.2, compute_angle_diff(player.pose.rotation, angle_to_ball)
    elif dist > 0.5:
        return 0.8, dribble_angle
    else:
        if not ball_is_in_front(player, ball):
            return 0.3, compute_angle_diff(player.pose.rotation, angle_to_ball)

        if teammates and (current_time - last_pass_time_yellow.get(idx, 0)) > 2.0:
            teammate = best_pass_teammate(player, teammates)
            if teammate:
                tx, ty = teammate.pose.position.x, teammate.pose.position.y
                angle_pass = math.atan2(ty - player.pose.position.y, tx - player.pose.position.x)
                ball.linear_speed = 5.0
                ball.pose.rotation = angle_pass
                last_pass_time_yellow[idx] = current_time
                return 1.5, compute_angle_diff(player.pose.rotation, angle_pass)

        target_x = 0.3
        target_y = (SCREEN_HEIGHT * PIX2M) / 2
        angle_goal = math.atan2(target_y - player.pose.position.y, target_x - player.pose.position.x)
        ball.linear_speed = 5.0
        ball.pose.rotation = angle_goal
        return 1.5, compute_angle_diff(player.pose.rotation, angle_goal)


def boundary_check(player, command):
    margin = 0.1
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


def main():
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Soccer Strategy with Passing")
    environment = Environment(window)

    players = [
        Player(Pose(3, 3, 0), 2, 2, 0.2),
        Player(Pose(6, 3, 0), 2, 2, 0.2),
        Player(Pose(3, 4, 0), 2, 2, 0.2),
        Player(Pose(6, 4, 0), 2, 2, 0.2)
    ]

    simulation = simulation2D(players, shockable=True, full_vision=False)
    avoid_radius = 0.3
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        ball = simulation.ball

        commands = []
        t = time.time()
        commands.append(boundary_check(players[0], red_strategy(players[0], ball, players[1], players[2], [players[2]], idx=0, current_time=t)))
        commands.append(boundary_check(players[1], yellow_strategy(players[1], ball, players[0], players[3], [players[3]], idx=1, current_time=t)))
        commands.append(boundary_check(players[2], red_strategy(players[2], ball, players[1], players[0], [players[0]], idx=2, current_time=t)))
        commands.append(boundary_check(players[3], yellow_strategy(players[3], ball, players[0], players[1], [players[1]], idx=3, current_time=t)))

        for i in range(len(players)):
            for j in range(i+1, len(players)):
                dx = players[i].pose.position.x - players[j].pose.position.x
                dy = players[i].pose.position.y - players[j].pose.position.y
                dist = math.hypot(dx, dy)
                if dist < avoid_radius:
                    players[i].pose.position.x += 0.01 * (dx / (dist + 1e-6))
                    players[i].pose.position.y += 0.01 * (dy / (dist + 1e-6))
                    players[j].pose.position.x -= 0.01 * (dx / (dist + 1e-6))
                    players[j].pose.position.y -= 0.01 * (dy / (dist + 1e-6))

        simulation.set_commands(commands)
        simulation.update()
        draw(simulation, window, environment)
        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()