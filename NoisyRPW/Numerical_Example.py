import NoisyRPW_Simulation
import Detection

# Define parameters
prompt = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
key = b'\xd3-\xf9\xf5\xbd\xf1\xdc\x0e\x1e.\x9a\xc2f\x17\x01\x18l6\x81\x1a\xdc4\xa2\xe1\x0f\x8b\xf8\xdd\x86\xd3P'
h = 3
beta = 0.5
delta = 5
n_values = range(1, 201)  
sigmas = [0, 0.25, 0.5, 0.75, 1.0]

# Generate and plot Z-scores for different sigma values
plt.clf()
plt.figure(constrained_layout=True)
plt.figure(figsize=(10, 6))
for sigma in sigmas:
    z_values = []
    for n in n_values:
        prompt_result = NoisyRPW.NoisyRPW(prompt, key, delta, h, n, sigma, beta)
        green_list_count = NoisyRPW.count_green_list_words(prompt, prompt_result, key, beta, h)
        z_scores = NoisyRPW.compute_z_score(green_list_count, beta, n)
        z_values.append(z_scores)
    plt.plot(n_values, z_values, label=f"sigma = {sigma}")

plt.title("Z-Score vs. n for Different Sigma Values")
plt.xlabel("n")
plt.ylabel("Z-Score")
plt.legend()
plt.grid(True)
plt.show()
