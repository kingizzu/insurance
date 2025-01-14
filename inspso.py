import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

# Automatically read the CSV file
data = pd.read_csv("insurance.csv")

# Encode categorical variables ('sex', 'smoker', 'region')
label_encoders = {
    'sex': LabelEncoder(),
    'smoker': LabelEncoder(),
    'region': LabelEncoder()
}

for column in ['sex', 'smoker', 'region']:
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features (X) and target (y)
X = data.drop('charges', axis=1)
y = data['charges']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fitness function
def fitness_function(solution):
    model = LinearRegression()
    model.coef_ = np.array(solution[:-1])
    model.intercept_ = solution[-1]
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    return mse

# PSO initialization functions
def initialize_particles(pop_size, dimensions, lower_bound, upper_bound):
    particles = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))
    velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (pop_size, dimensions))
    return particles, velocities

def update_velocity(velocities, particles, personal_best_positions, global_best_position, inertia, cognitive, social):
    r1 = np.random.random(size=particles.shape)
    r2 = np.random.random(size=particles.shape)
    cognitive_component = cognitive * r1 * (personal_best_positions - particles)
    social_component = social * r2 * (global_best_position - particles)
    return inertia * velocities + cognitive_component + social_component

def update_position(particles, velocities, lower_bound, upper_bound):
    particles += velocities
    return np.clip(particles, lower_bound, upper_bound)

def particle_swarm_optimization(pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social):
    particles, velocities = initialize_particles(pop_size, dimensions, lower_bound, upper_bound)
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p) for p in particles])

    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    history = []
    start_time = time.time()

    for generation in range(max_generations):
        velocities = update_velocity(velocities, particles, personal_best_positions, global_best_position, inertia, cognitive, social)
        particles = update_position(particles, velocities, lower_bound, upper_bound)

        current_scores = np.array([fitness_function(p) for p in particles])

        for i in range(pop_size):
            if current_scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = current_scores[i]
                personal_best_positions[i] = particles[i]

        if np.min(current_scores) < global_best_score:
            global_best_score = np.min(current_scores)
            global_best_position = particles[np.argmin(current_scores)]

        history.append(global_best_score)
        if generation % 10 == 0 or generation == max_generations - 1:
            st.write(f"Generation {generation}: Best Fitness (MSE) = {global_best_score}")

    elapsed_time = time.time() - start_time
    return global_best_position, global_best_score, history, elapsed_time

# Streamlit Interface
st.title("Insurance Charges Prediction using PSO")

# PSO Parameters
pop_size = st.number_input("Population Size", min_value=10, value=50)
dimensions = X_train.shape[1] + 1  # Number of features + 1 for intercept
max_generations = st.number_input("Maximum Generations", min_value=10, value=100)
lower_bound = st.number_input("Lower Bound", value=-1.0)
upper_bound = st.number_input("Upper Bound", value=1.0)
inertia = st.number_input("Inertia Weight", value=0.7)
cognitive = st.number_input("Cognitive Coefficient", value=1.5)
social = st.number_input("Social Coefficient", value=1.5)

if st.button("Run Optimization"):
    st.write("Running Particle Swarm Optimization...")

    best_solution, best_fitness, history, elapsed_time = particle_swarm_optimization(
        pop_size, dimensions, lower_bound, upper_bound, max_generations, inertia, cognitive, social
    )

    # Validate the best solution
    if best_solution is None or len(best_solution[:-1]) != X_train.shape[1]:
        st.error("Error: PSO failed to produce a valid solution. Check your parameters.")
    else:
        st.success(f"Optimization Completed! Best Fitness (MSE): {best_fitness}")
        st.write("Best Solution (Optimized Model Parameters):", best_solution)

        # Plot convergence
        st.line_chart(history)

        # Final model evaluation
        model = LinearRegression()
        model.coef_ = best_solution[:-1]
        model.intercept_ = best_solution[-1]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"Final Model Accuracy (MSE on Test Set): {mse}")
        st.write(f"Computational Efficiency: Time Taken = {elapsed_time:.2f} seconds")
