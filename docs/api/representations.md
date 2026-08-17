# `nssc.representations`

### `nssc.representations`

Encoders and decoders mapping observations x ∈ R^D ↔ latent z ∈ R^d.

Importing this package populates the ``ENCODERS`` / ``DECODERS`` registries.

#### `build_decoder(key: 'str', latent_dim: 'int', obs_dim: 'int', **kwargs: 'Any') -> 'Decoder'`

#### `build_encoder(key: 'str', obs_dim: 'int', latent_dim: 'int', **kwargs: 'Any') -> 'Encoder'`

### `nssc.representations.base`

Base classes for encoders and decoders.

Shape convention: observations ``x`` are ``(batch, time, obs_dim)``; latents
``z`` are ``(batch, time, latent_dim)``. Encoders may be causal (depend on
``x_{≤t}``) or pointwise; ``is_causal`` documents which. Non-neural encoders
(PCA) still subclass ``nn.Module`` so they serialise through ``state_dict``.

#### class `Decoder(latent_dim: 'int', obs_dim: 'int') -> 'None'`

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
- `forward(self, z: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.
- `num_parameters(self) -> 'int'`

#### class `Encoder(obs_dim: 'int', latent_dim: 'int') -> 'None'`

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

- `encode(self, x: 'Tensor') -> 'Tensor'`
- `fit(self, x: 'Tensor') -> 'None'`
- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.
- `num_parameters(self) -> 'int'`

### `nssc.representations.linear`

Linear (single ``nn.Linear``) encoder/decoder — the linear autoencoder baseline.

#### class `LinearDecoder(latent_dim: 'int', obs_dim: 'int', bias: 'bool' = True, **_: 'object') -> 'None'`

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

- `forward(self, z: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `LinearEncoder(obs_dim: 'int', latent_dim: 'int', bias: 'bool' = True, **_: 'object') -> 'None'`

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

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

### `nssc.representations.mlp`

Pointwise MLP encoder/decoder.

#### class `MLPDecoder(latent_dim: 'int', obs_dim: 'int', hidden_dims: 'Sequence[int]' = (128, 128), activation: 'str' = 'gelu', layernorm: 'bool' = False, dropout: 'float' = 0.0, **_: 'object') -> 'None'`

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

- `forward(self, z: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `MLPEncoder(obs_dim: 'int', latent_dim: 'int', hidden_dims: 'Sequence[int]' = (128, 128), activation: 'str' = 'gelu', layernorm: 'bool' = False, dropout: 'float' = 0.0, **_: 'object') -> 'None'`

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

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### `build_mlp(in_dim: 'int', out_dim: 'int', hidden_dims: 'Sequence[int]' = (128, 128), activation: 'str' = 'gelu', layernorm: 'bool' = False, dropout: 'float' = 0.0) -> 'nn.Sequential'`

``in_dim -> hidden... -> out_dim`` MLP applied to the last axis.

#### `get_activation(name: 'str') -> 'nn.Module'`

### `nssc.representations.multiscale`

Multi-scale (slow / fast) causal encoder.

The latent is split as ``z = [z_slow, z_fast]``. ``z_fast`` comes from a base
causal encoder on the raw input. ``z_slow`` comes from a separate small causal
encoder that sees only a *causally smoothed and strided* version of the input:
a left-padded moving average of window ``slow_window``, sampled every
``slow_window`` steps (at block starts, so nothing from the future leaks in) and
zero-order-hold upsampled back to length ``T``. ``z_slow`` is therefore
piecewise-constant over blocks and varies slowly by construction.

Both branches are causal, so the whole encoder is causal.

#### class `MultiScaleEncoder(obs_dim: 'int', latent_dim: 'int', slow_dim: 'int | None' = None, slow_window: 'int' = 8, base: 'str' = 'tcn', base_kwargs: 'dict[str, Any] | None' = None, slow_encoder: 'str' = 'tcn', slow_kwargs: 'dict[str, Any] | None' = None, **_: 'object') -> 'None'`

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

- `encode_fast(self, x: 'Tensor') -> 'Tensor'`
- `encode_slow(self, x: 'Tensor') -> 'Tensor'`
- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.
- `split(self, z: 'Tensor') -> 'tuple[Tensor, Tensor]'` — ``z`` (B,T,d) -> (z_slow (B,T,slow_dim), z_fast (B,T,fast_dim)).

#### `causal_moving_average(x: 'Tensor', window: 'int') -> 'Tensor'`

Left-padded moving average over time. ``x``: (B,T,D) -> (B,T,D).

#### `hold_upsample(x: 'Tensor', stride: 'int', length: 'int') -> 'Tensor'`

Zero-order-hold upsample (B,T',D) -> (B,length,D).

#### `stride_and_hold(x: 'Tensor', stride: 'int') -> 'tuple[Tensor, int]'`

Sample every ``stride``-th step (starting at t=0). Returns (x_strided, T).

### `nssc.representations.pca`

PCA encoder/decoder (closed-form linear subspace baseline).

``PCAEncoder.fit(x)`` computes the mean and top-``latent_dim`` principal
components of the flattened ``(B*T, D)`` observations via ``torch.linalg.svd``.
Forward is ``(x - mean) @ components.T``. It has no trainable parameters; the
fitted statistics live in buffers so they serialise through ``state_dict``.

#### class `PCADecoder(latent_dim: 'int', obs_dim: 'int', **_: 'object') -> 'None'`

Linear inverse of :class:`PCAEncoder`: ``x̂ = z @ components + mean``.

- `forward(self, z: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.
- `tie(self, encoder: 'PCAEncoder') -> 'PCADecoder'`

#### class `PCAEncoder(obs_dim: 'int', latent_dim: 'int', center: 'bool' = True, **_: 'object') -> 'None'`

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

- `fit(self, x: 'Tensor') -> 'None'` — Fit mean + components on ``x`` of shape (B,T,D) or (N,D).
- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### `explained_variance_curve(x: 'Tensor') -> 'Tensor'`

Cumulative explained-variance ratio per number of components.

``x``: (B,T,D) or (N,D). Returns a tensor of shape (D,) whose k-th entry is
the fraction of variance captured by the top-(k+1) principal components.

### `nssc.representations.rnn`

Recurrent (GRU / LSTM) causal encoders with a linear output projection.

#### class `GRUEncoder(obs_dim: 'int', latent_dim: 'int', hidden: 'int' = 64, n_layers: 'int' = 1, dropout: 'float' = 0.0, **_: 'object') -> 'None'`

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


#### class `LSTMEncoder(obs_dim: 'int', latent_dim: 'int', hidden: 'int' = 64, n_layers: 'int' = 1, dropout: 'float' = 0.0, **_: 'object') -> 'None'`

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


### `nssc.representations.ssm`

Lightweight diagonal linear state-space encoder (S4D / Mamba-lite style).

Each layer holds, per model channel ``h`` (``d_model`` channels) and per state
``n`` (``d_state`` states), a *real diagonal* continuous-time system

    ẋ = A x + B u,   y = C x + D u,   A = -exp(log_a) < 0   (stable by construction)

discretised with zero-order hold at a learnable step Δ (per channel):

    Ā = exp(Δ A),   B̄ = (Ā - 1) / A · B .

The recurrence ``x_t = Ā x_{t-1} + B̄ u_t`` is evaluated with a *chunked scan*:
inside each chunk of length ``L`` the response is computed in parallel from the
materialised powers ``Ā^k`` (a small ``L×L`` lower-triangular kernel), and the
chunk-final states are carried sequentially across ``T / L`` chunks. This is
exactly causal: ``y_t`` depends only on ``u_{≤t}``.

Each SSM layer is followed by GLU mixing and wrapped in a pre-LayerNorm
residual block; ``n_layers`` blocks are stacked; a linear head projects to
``latent_dim``.

#### class `DiagonalSSM(d_model: 'int', d_state: 'int' = 16, dt_min: 'float' = 0.001, dt_max: 'float' = 0.1, chunk: 'int' = 32) -> 'None'`

Per-channel real-diagonal SSM layer, (B,T,H) -> (B,T,H).

- `discretize(self) -> 'tuple[Tensor, Tensor]'`
- `forward(self, u: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `SSMBlock(d_model: 'int', d_state: 'int', expand: 'int', dropout: 'float', chunk: 'int') -> 'None'`

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

- `forward(self, h: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `SSMEncoder(obs_dim: 'int', latent_dim: 'int', d_model: 'int' = 64, d_state: 'int' = 16, n_layers: 'int' = 2, expand: 'int' = 2, dropout: 'float' = 0.0, chunk: 'int' = 32, **_: 'object') -> 'None'`

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

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### `chunked_diagonal_scan(a_bar: 'Tensor', bu: 'Tensor', chunk: 'int' = 32) -> 'Tensor'`

Causal linear recurrence ``x_t = a_bar * x_{t-1} + bu_t`` with constant diagonal ``a_bar``.

``a_bar``: (H,N) in (0,1);  ``bu``: (B,T,H,N).  Returns ``x``: (B,T,H,N).

### `nssc.representations.tcn`

Causal dilated temporal convolutional encoder (TCN).

Each residual block: LeftPad -> Conv1d(dilation) -> act -> Conv1d(k=1) -> +skip.
Left padding only, so ``z_t`` depends on ``x_{≤t}``. Receptive field is
``1 + (k-1) * sum(dilations)``.

#### class `CausalConv1d(in_ch: 'int', out_ch: 'int', kernel_size: 'int', dilation: 'int' = 1) -> 'None'`

1D convolution over (B,C,T) with left-only padding (causal).

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `TCNBlock(channels: 'int', kernel_size: 'int', dilation: 'int', activation: 'str', dropout: 'float') -> 'None'`

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

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.

#### class `TemporalConvEncoder(obs_dim: 'int', latent_dim: 'int', channels: 'int' = 64, kernel_size: 'int' = 3, n_layers: 'int' = 4, dilations: 'Sequence[int] | None' = None, activation: 'str' = 'gelu', dropout: 'float' = 0.0, **_: 'object') -> 'None'`

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

- `forward(self, x: 'Tensor') -> 'Tensor'` — Define the computation performed at every call.
