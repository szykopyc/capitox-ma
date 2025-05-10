import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
S0 = 92.23           # Current stock price
K = 144.01           # Conversion price
mu = 0.15            # Expected annual return
sigma = 0.45         # Annual volatility
r = 0.05             # Risk-free rate
T = 0.5              # Time to maturity in years (6 months)
n_simulations = 10000
dt = 1 / 252         # Daily time step
n_steps = int(T / dt)

# Convertible bond details
principal = 920_000_000
conversion_ratio = 6.9440
shares_issued = (principal / 1000) * conversion_ratio

# Monte Carlo simulation
final_prices = []
for _ in range(n_simulations):
    prices = [S0]
    for _ in range(n_steps):
        Z = np.random.normal()
        dS = prices[-1] * (r * dt + sigma * np.sqrt(dt) * Z)
        prices.append(prices[-1] + dS)
    final_prices.append(prices[-1])

final_prices = np.array(final_prices)
converted = final_prices > K
conversion_prob = np.mean(converted)

# Financial outcomes
expected_dilution = conversion_prob * shares_issued
expected_cash_out = (1 - conversion_prob) * principal
expected_equity_cost = (conversion_prob * shares_issued * 
                        np.mean(final_prices[converted]) if np.any(converted) else 0)

# Confidence Interval
ci_low, ci_high = np.percentile(final_prices, [2.5, 97.5])

# Output
print(f"Probability of Conversion: {conversion_prob:.2%}")
print(f"Expected Dilution (Shares): {expected_dilution:,.0f}")
print(f"Expected Cash Repayment: ${expected_cash_out:,.0f}")
print(f"Expected Equity Cost: ${expected_equity_cost:,.0f}")
print(f"95% Confidence Interval for Final Price: ${ci_low:.2f} – ${ci_high:.2f}")

# Histogram Plot
plt.figure(figsize=(10, 5))
plt.hist(final_prices, bins=50, color='skyblue', edgecolor='black')
plt.axvline(K, color='red', linestyle='--', label='Conversion Price')
plt.title('Distribution of Shopify Stock Price at Maturity (2025)')
plt.xlabel('Stock Price')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_price_histogram.png")
plt.show()

# CDF Plot
plt.figure(figsize=(10, 5))
sorted_prices = np.sort(final_prices)
cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
plt.plot(sorted_prices, cdf, label='CDF', color='green')
plt.axvline(K, color='red', linestyle='--', label='Conversion Price')
plt.title('CDF of Final Stock Price at Maturity')
plt.xlabel('Stock Price')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_price_cdf.png")
plt.show()