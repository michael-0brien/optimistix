import abc
from collections.abc import Callable
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from ._custom_types import Y
from ._misc import max_norm


_F = TypeVar("_F")


class AbstractConvergence(eqx.Module, Generic[Y]):
    """The abstract base class for convergence criteria. A convergence criterion
    consumes the change in the iterate and the change in the function value between
    two accepted steps of a solve, then checks whether the solve has converged.

    Minimisers and least-squares solvers each hold a convergence criterion as their
    `convergence` attribute.
    """

    norm: AbstractVar[Callable[[PyTree], Scalar]]

    @abc.abstractmethod
    def check(self, y: Y, y_diff: Y, f: _F, f_diff: _F) -> Bool[Array, ""]:
        """Decides whether the solve has converged.

        **Arguments:**

        - `y`: the value of the most recently accepted iterate.
        - `y_diff`: the difference between `y` and the previously accepted iterate.
        - `f`: the value of the function (or of the residuals, for a least-squares
            problem) at `y`.
        - `f_diff`: the difference between `f` and its value at the previously
            accepted iterate.

        **Returns:**

        A boolean scalar, which is `True` if the solve has converged.
        """


class CauchyConvergence(AbstractConvergence[Y]):
    """Converged if there is a small difference in both `y` space and `f` space, as
    determined by `rtol` and `atol`.

    Specifically, this checks that `y_diff < atol + rtol * y` and
    `f_diff < atol + rtol * f`, as measured by `norm`, and reports convergence when
    both of these are true.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm

    def check(self, y: Y, y_diff: Y, f: _F, f_diff: _F) -> Bool[Array, ""]:
        y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
        f_scale = (self.atol + self.rtol * ω(f).call(jnp.abs)).ω
        y_converged = self.norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
        f_converged = self.norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
        return y_converged & f_converged


CauchyConvergence.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
