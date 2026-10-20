"""Compatibility imports; implementation lives in the installable package."""
import _adapter_package  # noqa: F401
from flagrt_adapter.policy import Capability, Environment, InputSpec, OperatorPolicy, SiluPolicy
from flagrt_adapter.policy import KNOWN_FAILED_IMAGES
