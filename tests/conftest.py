"""Shared pytest configuration and hypothesis profiles."""

from hypothesis import HealthCheck, Verbosity, settings

# --- Hypothesis profiles ---

# dev: fast feedback during development
settings.register_profile(
    "dev",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
)

# ci: thorough checking
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
)

# debug: verbose output for investigating failures
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    suppress_health_check=[HealthCheck.too_slow],
)

# Default to dev profile for fast iteration
settings.load_profile("dev")
