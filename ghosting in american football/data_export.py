
import save_simulation_results from save_simulation_results
simulation_name="iteration_1"

initial_positions_attack = [(0, 0), (0, 3), (0, 10)]  
initial_positions_def = [(2, 3), (3, 5), (2, 15)]  
target_position = np.array([120, 53])  

# Velocities of attacking and defending players
velocities_attack = np.array([[1, 0], [0.5, 0], [1, 1]])  
velocities_def = np.array([[2, 1], [1, 1], [1, 2]])  

attacker_positions, defender_positions= sim_functions.simulate_play_rev(initial_positions_attack, target_position, velocities_attack,
                                                             initial_positions_def, velocities_def, time_step, total_time)


save_simulation_results(attacker_positions, defender_positions, simulation_name)