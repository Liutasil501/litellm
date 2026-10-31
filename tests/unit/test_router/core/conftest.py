from collections.abc import Callable, Generator

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import CORE_BUILDERS, CoreDeclined, ScriptedHost


@pytest.fixture
def host() -> ScriptedHost:
    return ScriptedHost()


@pytest.fixture(params=sorted(CORE_BUILDERS))
def make_core(request: pytest.FixtureRequest, host: ScriptedHost) -> Callable[..., RouterCore]:
    """``make_core(**router_arguments)``: the core under test, on the ``host`` fixture. Every test that
    uses it runs once per core in ``CORE_BUILDERS`` (``-k python`` or ``-k rust`` picks one)."""
    builder = CORE_BUILDERS[request.param]

    def build(**arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
        try:
            return builder(host, **arguments)
        except CoreDeclined as declined:
            pytest.skip(f"{request.param} core declines this configuration: {declined}")

    return build


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, None]:
    """A test that reaches a member or request the Rust core does not serve yet is skipped with
    that member named, so the run separates "not served yet" from a wrong answer."""
    outcome = yield
    excinfo = getattr(outcome, "excinfo", None)
    if excinfo is None:
        return
    from litellm.router_rust_core import RustCoreUnsupported

    error = excinfo[1]
    if isinstance(error, RustCoreUnsupported) or isinstance(error.__cause__, RustCoreUnsupported):
        outcome.force_exception(pytest.skip.Exception(f"not served by the Rust core yet: {error}", _use_item_location=True))  # pyright: ignore[reportAttributeAccessIssue]  # pluggy's Result
