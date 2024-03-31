# Noisy-Robust-Private-Watermarking
## Overview
NoisyRPW is a Python implementation of a Noisy Robust Private Watermarking algorithm designed to generate text data with a private watermark while preserving privacy of the watermarking rules against attackers by adding noise.

## Features
- Use hash function to simulate large language model when generating logit vector over the vocabulary.
- Add controlled noise to ensure differential privacy preservation.
- Embeds private watermarks in generated text data through cryptographic hashing and HMAC techniques.
- Subsample to make the detection robust against attacks.
- Flexible parameters for customization.
