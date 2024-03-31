import NoisyRPW_Simulation
import Detection
import secrets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Define parameters
prompt = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
key = b'\xd3-\xf9\xf5\xbd\xf1\xdc\x0e\x1e.\x9a\xc2f\x17\x01\x18l6\x81\x1a\xdc4\xa2\xe1\x0f\x8b\xf8\xdd\x86\xd3P'
h = 3
beta = 0.5
delta = 5
n_values = range(1, 201)  
sigmas = [0, 0.25, 0.5, 0.75, 1.0]



# Generate and plot Z-scores and n for different sigma values
plt.clf()
plt.figure(constrained_layout=True)
plt.figure(figsize=(10, 6))
for sigma in sigmas:
    z_values = []
    for n in n_values:
        prompt_result = NoisyRPW_Simulation.NoisyRPW(prompt, key, delta, h, n, sigma, beta)
        green_list_count = Detection.count_green_list_words(prompt, prompt_result, key, beta, h)
        z_scores = Detection.compute_z_score(green_list_count, beta, n)
        z_values.append(z_scores)
    plt.plot(n_values, z_values, label=f"sigma = {sigma}")

plt.title("Z-Score vs. n for Different Sigma Values")
plt.xlabel("n")
plt.ylabel("Z-Score")
plt.legend()
plt.grid(True)
plt.show()



# Plot trade-off funcyion for different n and sigma values
def calculate_mu(n, h, sigma):
    mu = np.sqrt(2 * n) / h * np.sqrt(np.exp(sigma ** (-2)) * norm.cdf(1.5 * sigma ** -1) + 3 * norm.cdf(-0.5 * sigma ** -1) - 2)
    return mu

def G_mu(mu, alpha):
    return norm.cdf(norm.ppf(1 - alpha) - mu)

n_values_sparse = [10, 50, 100, 200]
alpha_values = np.linspace(0.001, 0.999, 100)

plt.clf()
plt.figure(constrained_layout=True)
plt.figure(figsize=(10, 6))

for n in n_values:
    for sigma in sigma_values:
        mu = calculate_mu(n, h, sigma)
        G_mu_values = G_mu(mu, alpha_values)
        plt.plot(alpha_values, G_mu_values, label=f'n={n}, sigma={sigma}')

plt.title("GDP for Different n and Sigma Values")
plt.xlabel("Type I Error")
plt.ylabel("Type II Error")
plt.legend()
plt.grid(True)
plt.show()


