# @Author: cberube
# @Date:   2021-07-26 17:07:38
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   cberube
# @Last modified time: 2021-07-26 17:07:31


import torch
import numpy as np
import matplotlib.pyplot as plt

from ipvae import IPVAE


model = IPVAE()


model = model.load_weights()

# Generate a synthetic decay
x = model.module.decode(torch.randn(2))
# Add synthetic noise to it
xn = x + 5*(torch.rand(20) - 0.5)

# Denoise decay with a forward pass
xp = model.forward(xn)

# Plot comparison
t = np.arange(0.12+0.02, 0.92, 0.04)  # the IRIS ELREC Pro windows
plt.plot(t, x.detach().numpy(), '--k', label="Ground truth")
plt.plot(t, xn.detach().numpy(), '.k', label="Noisy input")
plt.plot(t, xp[0].detach().numpy(), '-C3', label="Denoised")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
plt.savefig("./figures/example.png", dpi=144, bbox_inches="tight")
