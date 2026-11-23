# Physics-Informed Neural Networks in MATLAB

Small, self-contained MATLAB examples of physics-informed neural networks (PINNs): networks trained to satisfy a differential equation, and optionally to fit measured data, using automatic differentiation in the MATLAB Deep Learning Toolbox.

## Examples

| Script | What it shows |
|---|---|
| `simple_pinn.m` | A minimal PINN that solves dy/dt = −y with y(0) = 1, using only the ODE residual and the initial condition. The result is compared with the exact solution e^(−t). |
| `ImprovePinnLossSystem.m` | A damped mass–spring oscillator, m·ẍ + μ·ẋ + k·x = 0. The network is trained on noisy measurements together with the physics residual. It includes time normalization, automatic balancing of the data and physics loss weights, gradient-norm monitoring, and live plots during training. |
| `free_ball_drag_F.m` | A reference numerical simulation (explicit time stepping) of a falling ball with linear drag, with peak detection on the trajectory. |

## Techniques

- `dlnetwork` and `dlfeval` with automatic differentiation to compute dx/dt and d²x/dt²
- Combined loss: data misfit + ODE residual + initial conditions
- Custom training loop with Adam (`adamupdate`)

## Requirements

MATLAB with the Deep Learning Toolbox. `ImprovePinnLossSystem.m` expects a data file, `massSpringDamperData.mat`, containing `tdata`, `xdata`, `tpinns`, `m`, `mu`, `k`, `t0` and `tmax`.
