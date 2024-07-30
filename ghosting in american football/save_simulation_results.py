import sim_functions	
import mysql.connector
import json

def save_simulation_results(attacker_positions, defender_positions, simulation_name):
    # Connect to the database
    try:
        conn = mysql.connector.connect(
            host="your_host",
            user="your_username",
            password="your_password",
            database="your_database"
        )
        cursor = conn.cursor()

        # Parse positions to JSON arrays
        attacker_positions_json = json.dumps(attacker_positions.tolist())
        defender_positions_json = json.dumps(defender_positions.tolist())

        # Call the stored procedure
        cursor.callproc("SaveSimulationResults", (attacker_positions_json, defender_positions_json, simulation_name))
        conn.commit()
        print("Simulation results saved successfully.")

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        # Close database connection
        if conn.is_connected():
            cursor.close()
            conn.close()

def save_simulation_outcomes_with_distance(distances, simulation_name):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host="your_host",
            user="your_username",
            password="your_password",
            database="your_database"
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # Call the stored procedure
        cursor.callproc('SaveSimulationOutcomes', (json.dumps(distances.tolist()), simulation_name))

        # Commit the transaction
        connection.commit()

        print("Simulation outcomes saved successfully.")

    except mysql.connector.Error as error:
        print("Error while saving simulation outcomes:", error)

    finally:
        # Close the cursor and connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()	