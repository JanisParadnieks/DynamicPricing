import pandas as pd
import numpy as np
import gym
from stable_baselines3 import DQN
import os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Data Preprocessing
df = pd.read_csv('/cleaned_output.csv', encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
# Remove all negative values (ignoring returns, discounts, etc.)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
# Drop irrelevant columns
df.drop(columns=['InvoiceNo', 'CustomerID', 'Country'], inplace=True)
# Create Date column
df['Date'] = df['InvoiceDate'].dt.date

# Group by StockCode, Date, UnitPrice
grouped_df = df.groupby(['StockCode', 'Date', 'UnitPrice'], as_index=False)['Quantity'].sum()

# Feature Engineering
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Date'] = df['InvoiceDate'].dt.date

# Aggregate data for all products together
daily_sales = df.groupby(['Date']).agg(
    Units_Sold=('Quantity', 'sum'),
    Avg_Price=('UnitPrice', 'mean')
).reset_index()

daily_sales['MA_3'] = daily_sales['Units_Sold'].rolling(3, min_periods=1).mean()
daily_sales['Price_Change'] = daily_sales['Avg_Price'].pct_change().fillna(0)

# Custom Gym Environment for Dynamic Pricing
class PricingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.day = 0
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.alpha = 0.1
        self.beta = 20.0

    def reset(self):
        self.day = 0
        return np.array([
            self.data['Avg_Price'].iloc[self.day],
            self.data['Units_Sold'].iloc[self.day],
            self.data['MA_3'].iloc[self.day]
        ], dtype=np.float32)

    def step(self, action):
        old_price = self.data['Avg_Price'].iloc[self.day]

        # Define price adjustment
        price_adjustment = [-0.02, -0.01, 0, 0.01, 0.02][action]
        new_price = old_price * (1 + price_adjustment)

        # Set a price floor
        min_price = old_price * 0.85  # Cannot drop below 85% of the average price
        if new_price < min_price:
            new_price = min_price  # Enforce price floor
            price_penalty = -50  # Apply penalty for hitting the price floor
        else:
            price_penalty = 0

        # Simulate demand elasticity
        elasticity = -1.5
        noise = np.random.normal(scale=0.1)
        demand = max(int(self.data['Units_Sold'].iloc[self.day] * (1 + elasticity * price_adjustment) + noise), 1)

        # Penalize low revenue
        revenue = new_price * self.data['Units_Sold'].iloc[self.day]
        min_revenue = old_price * self.data['Units_Sold'].iloc[self.day] * 0.75
        revenue_penalty = -100 if revenue < min_revenue else 0

        # Modify Reward Calculation
        stability_bonus = -abs(price_adjustment) ** 2 * 5
        total_reward = revenue + stability_bonus + revenue_penalty

        self.day += 1
        done = self.day >= len(self.data)


        if not done:
            next_state = np.array([
                new_price,
                demand,
                self.data['MA_3'].iloc[self.day]
            ], dtype=np.float32)
        else:
            next_state = self.reset()

        return next_state, total_reward, done, {}

# Initialize RL environment
env = PricingEnv(daily_sales)

# Static Pricing Revenue and Demand Calculation
daily_sales['Static_Revenue'] = daily_sales['Avg_Price'] * daily_sales['Units_Sold']
static_total_revenue = daily_sales['Static_Revenue'].sum()
print(f" Static Pricing Total Revenue: {static_total_revenue:.2f}") #20 016 275.75
static_demand = daily_sales['Units_Sold'].sum()


# Train RL model
model_path = "dqn_dynamic_pricing_model.zip"
if os.path.exists(model_path):
    print("Loading existing model...")
    model = DQN.load(model_path, env=env)
else:
    print("No existing model found. Training from scratch...")
    model = DQN('MlpPolicy', env, learning_rate=0.005, buffer_size=10000, #might be a good idea to increase buffer
                exploration_fraction=0.8, exploration_final_eps=0.2, verbose=1)

print("Evaluating RL performance BEFORE training-")
obs = env.reset()
pre_train_revenue = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, revenue, done, _ = env.step(action)
    pre_train_revenue += revenue
print(f"Total Revenue BEFORE training: {pre_train_revenue:.2f}")

model.learn(total_timesteps=10000) # changeable (last run was 10 000 000 steps which took almost 4 hours)
model.save(model_path)

print("Evaluating RL performance AFTER training-")
obs = env.reset()
post_train_revenue = 0
done = False
actions_list = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_list.append(action)
    obs, revenue, done, _ = env.step(action)
    post_train_revenue += revenue

print(f"Total Revenue AFTER training: {post_train_revenue:.2f}")

# Evaluation if steps are taken at all
obs = env.reset()
price_history, demand_history, revenue_history = [], [], []
total_adjusted_price, total_demand = 0, 0


done = False
actions_list = []
while not done:

    action, _ = model.predict(obs, deterministic=True)
    print(f"RL action: {action}")
    obs, revenue, done, _ = env.step(action)

    price_history.append(obs[0])
    demand_history.append(obs[1])
    revenue_history.append(revenue)

    total_adjusted_price += obs[0]
    total_demand += obs[1]

# Print Summation Check
print(f"Total Adjusted Price Sum: {total_adjusted_price:.2f}")
print(f"Total Demand Sum: {total_demand}")

# Compare RL Demand vs. Static Demand**
print(f"Static Pricing Total Demand: {static_demand}")
print(f"RL Pricing Total Demand: {total_demand}")

# RL Total Revenue
total_rl_revenue = sum(revenue_history)
print(f"Total Revenue Across All Products: {total_rl_revenue:.2f}")

# Prepare trimmed dataset
daily_sales_trimmed = daily_sales.iloc[:len(revenue_history)].copy()
daily_sales_trimmed['RL_Revenue'] = revenue_history
daily_sales_trimmed['Date'] = pd.to_datetime(daily_sales_trimmed['Date'])

# 2. Sort by date
daily_sales_trimmed = daily_sales_trimmed.sort_values('Date').reset_index(drop=True)

# Create 20-day period labels
period_length = 20
daily_sales_trimmed['Period'] = (daily_sales_trimmed.index // period_length) + 1
daily_sales_trimmed['Period_Label'] = 'Days ' + (
    (daily_sales_trimmed.index // period_length) * period_length + 1
).astype(str) + '-' + (
    (daily_sales_trimmed.index // period_length + 1) * period_length
).astype(str)

# Group by 20-day periods
periodic_revenue = daily_sales_trimmed.groupby('Period_Label').agg({
    'Static_Revenue': 'sum',
    'RL_Revenue': 'sum'
}).reset_index()

# Create bar chart
fig = px.bar(
    periodic_revenue,
    x='Period_Label',
    y=['Static_Revenue', 'RL_Revenue'],
    title="20-Day Revenue Comparison: Static vs RL Pricing",
    labels={'value': 'Total Revenue', 'Period_Label': 'Period', 'variable': 'Strategy'},
    barmode='group'
)
fig.show()
 # Add columns for visualizations
daily_sales_trimmed['RL_Demand'] = demand_history
daily_sales_trimmed['RL_Price'] = price_history
daily_sales_trimmed['Cumulative_Static_Revenue'] = daily_sales_trimmed['Static_Revenue'].cumsum()
daily_sales_trimmed['Cumulative_RL_Revenue'] = daily_sales_trimmed['RL_Revenue'].cumsum()

# Cumulative Revenue Over Time 
plt.figure(figsize=(12, 5))
plt.plot(daily_sales_trimmed['Date'], daily_sales_trimmed['Cumulative_Static_Revenue'], label='Static Pricing')
plt.plot(daily_sales_trimmed['Date'], daily_sales_trimmed['Cumulative_RL_Revenue'], label='RL Pricing')
plt.title("Cumulative Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Daily Demand Comparison
plt.figure(figsize=(12, 5))
plt.plot(daily_sales_trimmed['Date'], daily_sales_trimmed['Units_Sold'], label='Static Demand')
plt.plot(daily_sales_trimmed['Date'], daily_sales_trimmed['RL_Demand'], label='RL Demand')
plt.title("Daily Demand Comparison: Static vs RL")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
