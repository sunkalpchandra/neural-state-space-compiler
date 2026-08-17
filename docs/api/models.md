# `nssc.models`

### `nssc.models`

Composite models bundling encoder, dynamics and decoder.

### `nssc.models.builder`

Build a LatentModel from a model config dict.

Config schema::

    latent_dim: 8
    encoder: {name: mlp, kwargs: {hidden_dims: [128, 128]}}
    decoder: {name: mlp, kwargs: {}}
    dynamics: {name: residual_mlp, kwargs: {hidden_dims: [128, 128]}}

#### `build_latent_model(cfg: 'dict[str, Any]', obs_dim: 'int') -> 'LatentModel'`

#### `model_name(cfg: 'dict[str, Any]') -> 'str'`

### `nssc.models.latent_model`

LatentModel: encoder + latent dynamics + decoder.

Provides the three evaluation modes used throughout the project:

* ``reconstruct(x)``           x̂_t = D(E(x))_t
* ``predict_teacher_forced(x)`` x̂_{t+1} = D(F(E(x)_t))   (one-step, ground truth latents)
* ``rollout(x_context, H)``     encode context, recurse F for H steps, decode  (recursive)

#### class `LatentModel(encoder: 'Encoder', dynamics: 'Dynamics', decoder: 'Decoder', config: 'dict[str, Any] | None' = None) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `decode(self, z: 'Tensor') -> 'Tensor'`
- `encode(self, x: 'Tensor') -> 'Tensor'`
- `latent_trajectory(self, x: 'Tensor') -> 'Tensor'`
- `num_parameters(self) -> 'dict[str, int]'`
- `predict_teacher_forced(self, x: 'Tensor') -> 'tuple[Tensor, Tensor, Tensor]'` — Returns (x̂_{2:T}, z_{1:T}, ẑ_{2:T}) with ẑ_{t+1} = F(z_t).
- `reconstruct(self, x: 'Tensor') -> 'Tensor'`
- `rollout(self, x_context: 'Tensor', horizon: 'int', u: 'Tensor | None' = None) -> 'tuple[Tensor, Tensor]'` — Encode context, take last latent, roll forward ``horizon`` steps.
- `rollout_from_latent(self, z0: 'Tensor', horizon: 'int', u: 'Tensor | None' = None) -> 'tuple[Tensor, Tensor]'`
