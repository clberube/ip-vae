# IP-VAE
PyTorch implementation of the time-domain induced polarization variational autoencoder.

Original paper:

## Setup

### Dependencies
- PyTorch
- NumPy
- matplotlib (optional)

### Installation
```console
python setup.py install
```

## Usage

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from ipvae import IPVAE

model = IPVAE()
model.load_weights()

# Generate an arbitrary noisy decay
x = torch.exp(torch.linspace(10, 1, 20)/5) + 5*(torch.rand(20)-0.5)

# Forward pass of the IP-VAE
xp = model.forward(x)

# Plot comparison
t = np.arange(0.12+0.02, 0.92, 0.04)  # the IRIS ELREC Pro windows
plt.plot(t, x.numpy(), 'o', label="Input")
plt.plot(t, xp[0].detach().numpy(), label="IP-VAE")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
plt.savefig("./figures/example.png")
```

![example](./figures/example.png)
