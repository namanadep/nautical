import streamlit as st
import numpy as np
import random
from geopy.distance import great_circle
import plotly.graph_objects as go
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Port coordinates
port_coordinates = {
    "Mumbai": (18.9438, 72.8387),
    "Cochin": (9.9667, 76.2667),
    "Rajula": (20.9201, 71.5095)
}

ports = list(port_coordinates.keys())
n_ports = len(ports)

distances = np.zeros((n_ports, n_ports))
for i in range(n_ports):
    for j in range(i + 1, n_ports):
        port1, port2 = ports[i], ports[j]
        coord1, coord2 = port_coordinates[port1], port_coordinates[port2]
        distance = great_circle(coord1, coord2).nautical
        distances[i][j] = distances[j][i] = distance

# Define weather penalty mapping
weather_penalty = {
    "Clear": 1.0,   # No penalty
    "Cloudy": 0.9,  # Slight penalty
    "Rainy": 0.7,   # Moderate penalty
    "Stormy": 0.4   # High penalty
}

# Ant Colony Optimization class
class AntColonyOptimization:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, evaporation_rate, fuel_efficiency, ship_speed, weather_conditions, route_costs):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.n_ports = len(distances)
        self.pheromones = np.ones((self.n_ports, self.n_ports))
        self.fuel_efficiency = fuel_efficiency
        self.ship_speed = ship_speed
        self.weather_conditions = weather_conditions
        self.route_costs = route_costs

    def run(self, progress_callback):
        best_path = None
        best_cost = float('inf')

        for iteration in range(self.n_iterations):
            paths = self.construct_paths()
            self.update_pheromones(paths)

            iteration_best_path = min(paths, key=lambda x: self.path_cost(x))
            iteration_best_cost = self.path_cost(iteration_best_path)

            if iteration_best_cost < best_cost:
                best_path = iteration_best_path
                best_cost = iteration_best_cost

            progress_callback(iteration, paths, best_path, best_cost, self.pheromones)

        return best_path, best_cost

    def construct_paths(self):
        return [self.construct_single_path() for _ in range(self.n_ants)]

    def construct_single_path(self):
        path = [0]
        while path[-1] != self.n_ports - 1:
            current = path[-1]
            next_port = self.choose_next_port(current, path)
            path.append(next_port)
        return path

    def choose_next_port(self, current, path):
        unvisited = set(range(self.n_ports)) - set(path)
        if not unvisited:
            return self.n_ports - 1
        probabilities = self.calculate_probabilities(current, unvisited)
        return random.choices(list(unvisited), weights=probabilities)[0]

    def calculate_probabilities(self, current, available):
        probabilities = []
        for port in available:
            pheromone = self.pheromones[current][port]
            distance = self.distances[current][port]
            weather_condition = self.weather_conditions[current][port]
            weather_factor = weather_penalty.get(weather_condition, 1.0)
            probability = (pheromone ** self.alpha) * ((1 / distance) ** self.beta) * weather_factor
            probabilities.append(probability)
        return probabilities

    def update_pheromones(self, paths):
        self.pheromones *= (1 - self.evaporation_rate)
        for path in paths:
            cost = self.path_cost(path)
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1 / cost
                self.pheromones[path[i + 1]][path[i]] += 1 / cost

    def path_cost(self, path):
        total_distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
        fuel_consumption = total_distance * self.fuel_efficiency
        travel_time = total_distance / self.ship_speed
        weather_cost = 1.0
        total_route_cost = 0.0
        for i in range(len(path) - 1):
            weather_condition = self.weather_conditions[path[i]][path[i + 1]]
            weather_cost *= weather_penalty.get(weather_condition, 1.0)
            total_route_cost += self.route_costs[path[i]][path[i + 1]]

        cost = (total_distance + fuel_consumption + travel_time + total_route_cost) * weather_cost
        return cost

# ANN training function from main.py
def train_ANN(historical_data):
    if historical_data.empty:
        st.error("No historical data available for training.")
        return None

    target_columns = ['Optimal Route', 'Route Cost', 'Fuel Consumption', 'Travel Time']
    feature_columns = [col for col in historical_data.columns if col not in target_columns]

    if not all(col in historical_data.columns for col in target_columns):
        st.error("Target columns are missing in the historical data.")
        return None

    X = historical_data[feature_columns]
    y = historical_data[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = layers.Sequential([
        layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(target_columns), activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    test_loss, test_mae = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    model.save('trained_ANN_model.h5')
    return model

def create_sea_route(start, middle, end):
    """
    Create a sea route with curvature that passes through a middle point.
    :param start: Start coordinates (lat, lon)
    :param middle: Middle point coordinates (lat, lon)
    :param end: End coordinates (lat, lon)
    :return: List of lat/lon points forming a curved route from start to middle, then middle to end.
    """
    def curve_between_points(p1, p2, curve_factor=0.2):
        """
        Create a curved route between two points by generating a midpoint adjusted for curvature.
        :param p1: Start point (lat, lon)
        :param p2: End point (lat, lon)
        :param curve_factor: Factor by which to curve the route
        :return: List of points [p1, midpoint, p2] for the curved route
        """
        lat1, lon1 = p1
        lat2, lon2 = p2

        # Calculate a midpoint between the two points
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2

        # Apply the curve factor
        delta_lat = (lat2 - lat1) * curve_factor
        mid_lon -= abs(delta_lat)

        return [p1, (mid_lat, mid_lon), p2]

    # First leg: start -> middle (Rajula -> Mumbai)
    leg1 = curve_between_points(start, middle)

    # Second leg: middle -> end (Mumbai -> Cochin)
    leg2 = curve_between_points(middle, end)

    # Combine the two legs into one route
    return leg1[:-1] + leg2  # Avoid duplicating the middle point

# Visualization of routes
def plot_route(path, port_coordinates, pheromones):
    """
    Plot the sea route on the map with curvature through Mumbai from Rajula to Cochin.
    :param path: List of port indices representing the route
    :param port_coordinates: Dictionary of port coordinates
    :param pheromones: Matrix of pheromones (to be used for route intensity, though not in use here)
    :return: None
    """
    fig = go.Figure()

    # Define the route: Rajula -> Mumbai -> Cochin
    start_port = port_coordinates["Rajula"]
    middle_port = port_coordinates["Mumbai"]
    end_port = port_coordinates["Cochin"]

    # Create a curved sea route passing through Mumbai
    route = create_sea_route(start_port, middle_port, end_port)
    lats, lons = zip(*route)

    # Plot the route
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=lons,
        lat=lats,
        line=dict(width=2, color="rgba(0, 255, 0, 0.6)")
    ))

    # Plot the ports
    port_lats, port_lons = zip(*port_coordinates.values())
    fig.add_trace(go.Scattermapbox(
        mode="markers+text",
        lon=port_lons,
        lat=port_lats,
        marker={'size': 10, 'color': 'blue'},
        text=list(port_coordinates.keys()),
        textposition="top center"
    ))

    # Configure map layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lon=np.mean(port_lons), lat=np.mean(port_lats)),
            zoom=5
        ),
        showlegend=False,
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        height=600
    )

    return fig


# Streamlit UI
st.title('Real-time Ant Colony Optimization and ANN for Ship Routing')
st.write("Finding optimal sea route from Rajula to Cochin")

n_ants = 20
n_iterations = 50
alpha = 1.00
beta = 2.00
evaporation_rate = 0.10

st.sidebar.header('Ship Parameters')
fuel_efficiency = st.sidebar.slider('Fuel Efficiency (liters per nautical mile)', 1.0, 10.0, 5.0)
ship_speed = st.sidebar.slider('Ship Speed (nautical miles per hour)', 10, 30, 20)

map_placeholder = st.empty()
info_placeholder = st.empty()

if st.button('Run Simulation'):
    try:
        # Load the historical data
        historical_data = pd.read_csv("E:/NAMAN/SIH Nautical/Datasets/Historical_data.csv")
        
        # Train the ANN on the historical data
        ANN_model = train_ANN(historical_data)
        
        # Assuming historical_data has 'Weather Conditions' and 'Route Cost' columns
        weather_conditions = np.zeros((n_ports, n_ports), dtype=object)
        route_costs = np.zeros((n_ports, n_ports))  # Initialize route cost array

        for _, row in historical_data.iterrows():
            start_port = row['Start Port']
            end_port = row['End Port']
            weather = row['Weather Conditions']
            route_cost = row['Route Cost']  # Get route cost from dataset

            if start_port in port_coordinates and end_port in port_coordinates:
                i, j = ports.index(start_port), ports.index(end_port)
                weather_conditions[i][j] = weather_conditions[j][i] = weather
                route_costs[i][j] = route_costs[j][i] = route_cost  # Set route cost

    except FileNotFoundError:
        st.error("Historical data file not found.")
        weather_conditions = np.zeros((n_ports, n_ports), dtype=object)
        route_costs = np.zeros((n_ports, n_ports))  # Initialize with zeros if data is missing

    # Create the ACO instance with weather and route costs
    aco = AntColonyOptimization(
        distances=distances,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        fuel_efficiency=fuel_efficiency,
        ship_speed=ship_speed,
        weather_conditions=weather_conditions,
        route_costs=route_costs
    )

    # Callback to update progress
    def progress_callback(iteration, paths, best_path, best_cost, pheromones):
        info_placeholder.write(f"Iteration {iteration}: Best Cost = {best_cost}")

    # Run the ACO simulation
    best_path, best_cost = aco.run(progress_callback)

    # Plot the optimized route
    fig = plot_route([best_path], port_coordinates, aco.pheromones)
    map_placeholder.plotly_chart(fig)
