import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
import gym


class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration-exploitation tradeoff
        
        # Initialize Q-table with zeros
        self.Q_table = np.zeros((state_space, action_space))

    def choose_action(self, state_index):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            action = np.random.choice(self.action_space)
        else:
            # Exploitation: choose the best action
            action = np.argmax(self.Q_table[int(state_index)])  # Cast state_index to int
        return action

    def update_Q_table(self, state_index, action, reward, next_state_index):
        if next_state_index < self.state_space:
            best_next_action = np.argmax(self.Q_table[int(next_state_index)])  # Cast next_state_index to int
            # Update rule
            self.Q_table[int(state_index), action] = (
                self.Q_table[int(state_index), action]
                + self.learning_rate
                * (reward + self.discount_factor * self.Q_table[int(next_state_index), best_next_action] - self.Q_table[int(state_index), action])
            )


class StockTradingEnv:
    def __init__(self, states, initial_balance=10000.0, trading_fee=0.001, action_space=3):
        self.states = states
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.action_space = action_space  # Buy, Sell, Hold
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        
        # Define the observation space
        self.observation_space = len(states.columns)
        
    class StockTradingEnv:
        def __init__(self, states, initial_balance=10000.0, trading_fee=0.001, action_space=3):
            self.states = states
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.action_space = action_space  # Buy, Sell, Hold
            self.current_step = 0
            self.balance = initial_balance
            self.position = 0
            
            # Define the observation space
            self.observation_space = len(states.columns)
        
    class StockTradingEnv:
        def __init__(self, states, initial_balance=10000.0, trading_fee=0.001, action_space=3):
            self.states = states
            self.initial_balance = initial_balance
            self.trading_fee = trading_fee
            self.action_space = action_space  # Buy, Sell, Hold
            self.current_step = 0
            self.balance = initial_balance
            self.position = 0
            
            # Define the observation space
            self.observation_space = len(states.columns)
        
    def step(self, action):
        current_state = self.states.iloc[self.current_step]
        
        reward = 0.0
        done = False
        
        # Use self.balance directly without float conversion
        if float(self.balance) > 0:  # Ensure balance is a scalar and works correctly
            shares_to_buy = int(self.balance / current_state['Close'])
            if shares_to_buy > 0:
                cost = shares_to_buy * current_state['Close']
                self.balance -= cost + (cost * self.trading_fee)
                self.position += shares_to_buy
        
        elif action == 1:  # Sell
            if self.position > 0:
                proceeds = self.position * current_state['Close']
                self.balance += proceeds - (proceeds * self.trading_fee)
                self.position = 0
        
        # Reward calculation
        reward = self.balance - self.initial_balance  # Remove float conversion
        
        done = self.current_step >= len(self.states) - 1
        
        self.current_step += 1
        return current_state.values, reward, done
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self.states.iloc[self.current_step].values

    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self.states.iloc[self.current_step].values


    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self.states.iloc[self.current_step].values



def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
    - data: DataFrame containing 'Close' prices
    - period: The period over which RSI is calculated
    
    Returns:
    - A DataFrame column with RSI values
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(data, ma_periods=[5, 20, 50]):
    """
    Calculate moving averages for different periods.
    
    Parameters:
    - data: DataFrame containing stock prices
    - ma_periods: List of moving average periods (e.g., [5, 20, 50])
    
    Returns:
    - A DataFrame with additional moving average columns
    """
    for period in ma_periods:
        data[f'ma_{period}'] = data['Close'].rolling(window=period).mean()
    return data

def prepare_states(data):
    """
    Prepare the state representation by calculating technical indicators.
    
    Parameters:
    - data: Raw stock data DataFrame
    
    Returns:
    - A DataFrame with selected state features
    """
    # Calculate moving averages
    data = calculate_moving_averages(data)
    
    # Calculate RSI
    data['rsi'] = calculate_rsi(data)
    
    # Define state features
    state_features = ['Close', 'Open', 'High', 'Low', 'Volume', 'ma_5', 'ma_20', 'ma_50', 'rsi']
    states = data[state_features].dropna()  # Drop rows with NaN values due to moving average/RSI calculations
    return states

ticker = 'TSLA'  # Change ticker as needed

try:
    # Fetch data for the maximum available range
    data = yf.download(ticker, period='max')
    if data.empty:
        print(f"No data found for {ticker}.")
    else:
        print(f"Data for {ticker} fetched successfully.")
        states = prepare_states(data)
except Exception as e:
    print(f"An error occurred: {e}")


# Initialize environment and agent
env = StockTradingEnv(states)
agent = QLearningAgent(state_space=states.shape[0], action_space=3)

# Training parameters
timesteps = 100000
for episode in range(timesteps):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(env.current_step)
        next_state, reward, done = env.step(action)
        agent.update_Q_table(env.current_step, action, reward, env.current_step + 1)
        total_reward += reward
    
    if episode % 1000 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

print("Training complete.")
