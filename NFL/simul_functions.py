import pandas as pd
import numpy as np



















# Function to calculate velocities based on current positions and target position
def calculate_velocities(positions, target_position, max_speed):
    target_direction = target_position - positions
    distance = np.linalg.norm(target_direction, axis=1)
    velocities = (max_speed / distance).reshape(-1, 1) * target_direction
    return velocities

# Function to update player positions
def update_positions(positions, velocities, time_step):
    new_positions = positions + velocities * time_step
    return new_positions

# Function to simulate football play
def simulate_play(initial_positions_attack, target_position, max_speed_attack, initial_positions_def, max_speed_def, time_step, total_time):
    num_players_attack = len(initial_positions_attack)
    num_players_def = len(initial_positions_def)
    
    num_steps = int(total_time / time_step)

    positions_attack = np.array(initial_positions_attack, dtype=float)
    positions_def = np.array(initial_positions_def, dtype=float)

    # Store positions at each time step
    all_positions_attack = [positions_attack.copy()]
    all_positions_def = [positions_def.copy()]

    # Simulate player movements
    for _ in range(num_steps):
        # Calculate velocities for attackers towards the target
        velocities_attack = calculate_velocities(positions_attack, target_position, max_speed_attack)

        # Calculate velocities for defenders towards the closest attacking player
        closest_attacker_index = np.argmin(np.linalg.norm(positions_def[:, np.newaxis] - positions_attack, axis=2), axis=1)
        target_positions_def = positions_attack[closest_attacker_index]
        velocities_def = calculate_velocities(positions_def, target_positions_def, max_speed_def)

        # Update positions
        positions_attack = update_positions(positions_attack, velocities_attack, time_step)
        positions_def = update_positions(positions_def, velocities_def, time_step)

        all_positions_attack.append(positions_attack.copy())
        all_positions_def.append(positions_def.copy())

    return np.array(all_positions_attack), np.array(all_positions_def)
	
	
	
	
def simulate_play_rev(initial_positions_attack, target_position, velocities_attack, initial_positions_def, velocities_def, time_step, total_time):
    num_players_attack = len(initial_positions_attack)
    num_players_def = len(initial_positions_def)
    
    num_steps = int(total_time / time_step)

    positions_attack = np.array(initial_positions_attack, dtype=float)
    positions_def = np.array(initial_positions_def, dtype=float)

    # Store positions at each time step
    all_positions_attack = [positions_attack.copy()]
    all_positions_def = [positions_def.copy()]

    # Simulate player movements
    for _ in range(num_steps):
        # Update positions
        positions_attack = update_positions(positions_attack, velocities_attack, time_step)
        positions_def = update_positions(positions_def, velocities_def, time_step)

        all_positions_attack.append(positions_attack.copy())
        all_positions_def.append(positions_def.copy())

    return np.array(all_positions_attack), np.array(all_positions_def)	
	
	
import numpy as np

# Function to calculate distances between all pairs of players
def calculate_pairwise_distances(positions):
    num_players = positions.shape[0]
    distances = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(num_players):
            distances[i, j] = np.linalg.norm(positions[i] - positions[j])
    return distances

# Function to update player positions
def update_positions(positions, velocities, time_step):
    new_positions = positions + velocities * time_step
    return new_positions

# Function to simulate football play
def simulate_play_with_dist(initial_positions_attack, target_position, velocities_attack, initial_positions_def, velocities_def, time_step, total_time):
    num_players_attack = len(initial_positions_attack)
    num_players_def = len(initial_positions_def)
    
    num_steps = int(total_time / time_step)

    positions_attack = np.array(initial_positions_attack, dtype=float)
    positions_def = np.array(initial_positions_def, dtype=float)

    # Store positions and distances at each time step
    all_positions_attack = [positions_attack.copy()]
    all_positions_def = [positions_def.copy()]
    all_distances = []

    # Simulate player movements
    for _ in range(num_steps):
        # Update positions
        positions_attack = update_positions(positions_attack, velocities_attack, time_step)
        positions_def = update_positions(positions_def, velocities_def, time_step)

        all_positions_attack.append(positions_attack.copy())
        all_positions_def.append(positions_def.copy())

        # Calculate distances between players
        all_positions = np.concatenate((positions_attack, positions_def))
        distances = calculate_pairwise_distances(all_positions)
        all_distances.append(distances)

    return np.array(all_positions_attack), np.array(all_positions_def), np.array(all_distances)

