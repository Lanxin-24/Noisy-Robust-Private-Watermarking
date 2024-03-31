# Noisy-Robust-Private-Watermarking
## Overview
Noisy Robust Private Watermarking(NoisyRPW) algorithm is designed to generate text data with a private watermark while preserving privacy of the watermarking rules against attackers by adding noise. This repository contains Python code for simulating the NoisyRPW algorithm and evaluating the watermark detection.

## Features
- Use hash function to simulate large language model when generating logit vector over the vocabulary.
- Add controlled noise to ensure differential privacy preservation.
- Embeds private watermarks in generated text data through cryptographic hashing and HMAC techniques.
- Subsample to make the detection robust against attacks.
- Flexible parameters for customization.

## Usage
### NoisyRPW Simulation
The `NoisyRPW_Simulation` module provides functions to simulate the NoisyRPW algorithm. Use the `NoisyRPW` function to generate tokens based on the prompt and provided parameters.
```python
import NoisyRPW_Simulation
prompt = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
key = b'your_secret_key'
h = 3
beta = 0.5
delta = 10
n_max = 200
sigma = 0.75
prompt_result = NoisyRPW_Simulation.NoisyRPW(prompt, key, delta, h, n, sigma, beta)
print(prompt_result)
```

### Detection
The `Detection` module provides functions to detect green list words and compute the z-score. Use the `count_green_list_words` function to count green list words and the `compute_z_score` function to calculate the z-score.
```python
import Detection

green_list_count = Detection.count_green_list_words(prompt, prompt_result, key, beta, h)
print("Green List Count:", green_list_count)

z_score = Detection.compute_z_score(green_list_count, beta, n)
print("Z-score:", z_score)
```
## Dependencies

This project relies on the following Python libraries:

- `numpy`: A powerful library for numerical computing in Python.
- `matplotlib`: A comprehensive library for creating static, animated, and interactive visualizations in Python.
- `scipy`: A library used for scientific computing and technical computing.

You can install these dependencies using the following command:

```bash
pip install numpy matplotlib scipy
