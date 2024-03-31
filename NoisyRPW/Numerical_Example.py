import NoisyRPW_Simulation
import Detection
import secrets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm



# Define parameters
prompt = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
key = secrets.token_bytes(32)
h = 3
beta = 0.5
delta = 10
n_max = 200
n_values = range(1, 201) 
sigmas = [0.25, 0.5, 0.75, 1.0]



fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Generate and plot Z-scores and n for different sigma values
ax1 = axes[0]
for sigma in sigmas:
    z_values = []
    for n in n_values:
        prompt_result = NoisyRPW_Simulation.NoisyRPW(prompt, key, delta, h, n, sigma, beta)
        green_list_count = Detection.count_green_list_words(prompt, prompt_result, key, beta, h)
        z_scores = Detection.compute_z_score(green_list_count, beta, n)
        z_values.append(z_scores)
    ax1.plot(n_values, z_values, label=f"sigma = {sigma}")

ax1.set_title("Z-Score vs. n for Different Sigma Values")
ax1.set_xlabel("n")
ax1.set_ylabel("Z-Score")
ax1.legend()
ax1.grid(True)



# Plot trade-off funcyion for different n and sigma values
def calculate_mu(n, h, sigma):
    mu = np.sqrt(2 * n) / h * np.sqrt(np.exp(sigma ** (-2)) * norm.cdf(1.5 * sigma ** (-1)) + 3 * norm.cdf(-0.5 * sigma ** (-1)) - 2)
    return mu

def G_mu(mu, alpha):
    return norm.cdf(norm.ppf(1 - alpha) - mu)

n_values_sparse = [10, 100, 200]
sigmas_sparse = [0.5, 1.0]
alpha_values = np.linspace(0.001, 0.999, 100)

ax2 = axes[1]
for n in n_values:
    for sigma in sigma_values:
        mu = calculate_mu(n, h, sigma)
        G_mu_values = G_mu(mu, alpha_values)
        ax2.plot(alpha_values, G_mu_values, label=f'n={n}, sigma={sigma}')

ax2.set_title("GDP for Different n and Sigma Values")
ax2.set_xlabel("Type I Error")
ax2.set_ylabel("Type II Error")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


