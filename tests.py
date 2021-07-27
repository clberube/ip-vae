# @Author: cberube
# @Date:   2021-07-26 17:07:38
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   cberube
# @Last modified time: 2021-07-26 17:07:31


import torch
import numpy as np
import matplotlib.pyplot as plt

import ipvae

model = ipvae.Net()
model = model.load_weights()

# Generate a synthetic decay
x = model.decode(torch.randn(2))
# Add synthetic noise to it
xn = x + 5*(torch.rand(20) - 0.5)

# Denoise decay with a forward pass
xp = model.forward(xn)

# Plot comparison
t = np.arange(0.12+0.02, 0.92, 0.04)  # the IRIS ELREC Pro windows
plt.figure()
plt.plot(t, x.detach().numpy(), '--k', label="Ground truth")
plt.plot(t, xn.detach().numpy(), '.k', label="Noisy input")
plt.plot(t, xp[0].detach().numpy(), '-C3', label="Denoised")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
# plt.savefig("./figures/example-1.png", dpi=144, bbox_inches="tight")


xp = [model.forward(xn)[0] for _ in range(100)]
xp = torch.stack(xp)
# Compute statistics
xp_avg = torch.mean(xp, dim=0)
xp_std = torch.std(xp, dim=0)

# Plot comparison
xp = [model.forward(xn)[0] for _ in range(100)]
xp = torch.stack(xp)
# Compute statistics
xp_avg = torch.mean(xp, dim=0)
xp_std = torch.std(xp, dim=0)

plt.figure()
plt.plot(t, x.detach().numpy(), '--k', label="Ground truth")
plt.plot(t, xn.detach().numpy(), '.k', label="Noisy input")
plt.plot(t, xp_avg.detach().numpy(), '-C3', label="Denoised")
plt.fill_between(t,
                 (xp_avg-2*xp_std).detach().numpy(),
                 (xp_avg+2*xp_std).detach().numpy(),
                 color='C3', alpha=0.2, label=r"$2\sigma$")
plt.legend()
plt.ylabel("Chargeability (mV/V)")
plt.xlabel("$t$ (s)")
# plt.savefig("./figures/example-2.png", dpi=144, bbox_inches="tight")

%timeit model.forward(xn)
