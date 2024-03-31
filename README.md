# Noisy-Robust-Private-Watermarking
## Overview
NoisyRPW is a Python implementation of a Noisy Robust Private Watermarking algorithm designed to generate text data with a private watermark while preserving privacy of the watermarking rules against attackers by adding noise.

## Features
- Embeds private watermarks in generated text data.
- Use hash function to simulate large language model when generating logit vector over the vocabulary.
- Robust against attacks by subsampling.
- Noisy watermarking ensures differential privacy preservation.
- Flexible parameters for customization.

## Installation
You can install NoisyRPW from PyPI using pip:

```bash
pip install NoisyRPW
