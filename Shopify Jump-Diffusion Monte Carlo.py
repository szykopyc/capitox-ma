import numpy as np

# === Basic Parameters ===
S0 = 92.23                      # Initial stock price
K = 144.01                      # Conversion price
mu = 0.15                       # Expected return
sigma = 0.45                    # Annual volatility
T = 0.5                         # Time to maturity (in years)
dt = 1 / 252                    # Daily time step
n_steps = int(T / dt)
n_simulations = 10000

# === Convertible Bond Details ===
principal = 920_000_000
conversion_ratio = 6.9440       # Shares per $1,000
shares_issued = (principal / 1000) * conversion_ratio

# === Jump-Diffusion Parameters ===
lambda_j = 1.0                  # Avg jumps per year
mu_j = -0.02                    # Mean jump size (log)
sigma_j = 0.1                   # Std dev of jump size

# === Monte Carlo Simulation ===
jd_final_prices = []

for _ in range(n_simulations):
    S = S0
    for _ in range(n_steps):
        Z = np.random.normal()
        dW = sigma * np.sqrt(dt) * Z
        jump = 0
        if np.random.rand() < lambda_j * dt:
            jump = np.random.normal(mu_j, sigma_j)
        dS = S * (mu * dt + dW + jump)
        S += dS
    jd_final_prices.append(S)

jd_final_prices = np.array(jd_final_prices)

# === Post-Simulation Analysis ===
converted = jd_final_prices > K
conversion_prob = np.mean(converted)
expected_dilution = conversion_prob * shares_issued
expected_cash_out = (1 - conversion_prob) * principal
expected_equity_cost = conversion_prob * shares_issued * np.mean(jd_final_prices[converted]) if np.any(converted) else 0
ci_low, ci_high = np.percentile(jd_final_prices, [2.5, 97.5])

# === Results ===
print("\n--- Jump-Diffusion Convertible Bond Simulation Results ---")
print(f"Conversion Probability: {conversion_prob:.2%}")
print(f"Expected Dilution (Shares): {expected_dilution:,.0f}")
print(f"Expected Cash Repayment: ${expected_cash_out:,.0f}")
print(f"Expected Equity Cost (if converted): ${expected_equity_cost:,.0f}")
print(f"95% Confidence Interval (Final Price): ${ci_low:.2f} – ${ci_high:.2f}")
