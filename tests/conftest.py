"""A real PostgreSQL for the persistence tests.

- TEST_DATABASE_URL set (CI: a postgres service container): that database, emptied before each test.
- otherwise, if initdb and pg_ctl are on PATH: a throwaway cluster in a temp directory (embedded).
- otherwise the tests that need it are skipped, unless REQUIRE_POSTGRES=1 (CI), where they fail.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = Path(__file__).resolve().parent.parent / "src" / "model" / "completed" / "trading_model_episode_30.h5"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pg_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not (initdb and pg_ctl):
        if os.environ.get("REQUIRE_POSTGRES") == "1":
            pytest.fail("REQUIRE_POSTGRES=1 but no TEST_DATABASE_URL and no initdb/pg_ctl on PATH")
        pytest.skip("no PostgreSQL available (set TEST_DATABASE_URL or put initdb/pg_ctl on PATH)")
    tmp = tempfile.mkdtemp(prefix="trader-pg-")
    data, port = os.path.join(tmp, "data"), _free_port()
    # output to a file, never to a pipe: the server started by pg_ctl would keep a pipe open forever
    with open(os.path.join(tmp, "setup.log"), "w") as out:
        quiet = {"stdout": out, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "check": True}
        subprocess.run([initdb, "-D", data, "-U", "postgres", "-A", "trust", "-E", "UTF8", "--no-instructions"], **quiet)
        subprocess.run([pg_ctl, "-D", data, "-o", f"-p {port} -c listen_addresses=127.0.0.1", "-l",
                        os.path.join(tmp, "log"), "-w", "start"], **quiet)
    try:
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run([pg_ctl, "-D", data, "-m", "fast", "-w", "stop"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def store(pg_url):
    from trader.store import Store

    s = Store(pg_url, connect_timeout=30)  # Windows can take seconds per connection
    with s.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS portfolio, equity, trades")
    s.init_schema()
    return s
