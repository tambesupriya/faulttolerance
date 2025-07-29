# Let's write the Python source code to a .py file for Supriya's online two-timescale hybrid heuristic approach

import random
import math
import numpy as np
import pandas as pd

# Constants and Parameters
SERVICES = 4
BUDGET = 2500
W_LATENCY = 0.9
W_AVAILABILITY = 0.1
SERVICE_COST = [30, 40, 50, 60]
LAT_LIMIT = [1000, 2000, 3000, 4000]  # in ms
POISSON_LAMBDA = 3  # mean edge failure rate

# Load Edge and User Data
edge_df = pd.read_csv("edgecbd.csv")
user_df = pd.read_csv("usercbd.csv")

# Initialization
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c * 1000  # in meters

def compute_latency_matrix(users, edges):
    latency = []
    for u in users:
        row = []
        for e in edges:
            dist = haversine_distance(u[0], u[1], e[0], e[1])
            row.append(dist)  # 1ms per km simplification
        latency.append(row)
    return latency

def compute_availability_score(requests, edge_status):
    availability = np.ones((len(requests), len(edge_status)))
    for i in range(len(edge_status)):
        if edge_status[i] == 0:
            availability[:, i] = 0
    return availability

# Service Placement (Large Timescale)
def service_placement(edges, latency_matrix, user_demands, budget):
    edge_services = {}
    cost_used = 0
    for e_id in range(len(edges)):
        edge_services[e_id] = []
        for s_id in range(SERVICES):
            score = 0
            for u_id, u_lat in enumerate(latency_matrix):
                if user_demands[u_id] == s_id:
                    lat = u_lat[e_id]
                    score += (W_LATENCY * (1 / (lat + 1)))
            if cost_used + SERVICE_COST[s_id] <= budget:
                edge_services[e_id].append(s_id)
                cost_used += SERVICE_COST[s_id]
    return edge_services

# Request Allocation (Short Timescale)
def allocate_requests(users, edges, latency_matrix, user_demands, edge_services, edge_failures):
    primary = [-1] * len(users)
    backup = [-1] * len(users)
    for u_id, demand in enumerate(user_demands):
        scores = []
        for e_id in range(len(edges)):
            if demand in edge_services[e_id]:
                if edge_failures[e_id] == 0:
                    lat = latency_matrix[u_id][e_id]
                    score = (W_LATENCY * (1 / (lat + 1))) + (W_AVAILABILITY * 1)
                    scores.append((score, e_id))
        scores.sort(reverse=True)
        if scores:
            primary[u_id] = scores[0][1]
            if len(scores) > 1:
                backup[u_id] = scores[1][1]
    return primary, backup

# Simulate Edge Failures using Poisson Distribution
def simulate_failures(num_edges):
    failure_flags = [0] * num_edges
    num_failures = np.random.poisson(POISSON_LAMBDA)
    failed_edges = random.sample(range(num_edges), min(num_failures, num_edges))
    for f in failed_edges:
        failure_flags[f] = 1
    return failure_flags

# Main Loop
def online_two_timescale_heuristic(time_slots=5, users_per_slot=200, edges_count=40):
    all_users = user_df.sample(users_per_slot * time_slots).values.tolist()
    all_edges = edge_df.sample(edges_count).values.tolist()

    slot_results = []

    for t in range(time_slots):
        print(f"\\n--- Time Slot {t+1} ---")
        current_users = all_users[t * users_per_slot: (t + 1) * users_per_slot]
        user_demands = [random.randint(0, SERVICES - 1) for _ in range(users_per_slot)]

        latency_matrix = compute_latency_matrix(current_users, all_edges)
        edge_services = service_placement(all_edges, latency_matrix, user_demands, BUDGET)
        edge_failures = simulate_failures(edges_count)
        primary, backup = allocate_requests(current_users, all_edges, latency_matrix, user_demands, edge_services, edge_failures)

        covered_requests = sum(1 for p in primary if p != -1)
        backup_assigned = sum(1 for b in backup if b != -1)
        print(f"Users: {users_per_slot}, Primary Covered: {covered_requests}, Backup Assigned: {backup_assigned}")

        slot_results.append({
            "primary": primary,
            "backup": backup,
            "services": edge_services,
            "failures": edge_failures,
            "covered": covered_requests,
            "backup_count": backup_assigned
        })
    return slot_results

# Run Heuristic
results = online_two_timescale_heuristic()
