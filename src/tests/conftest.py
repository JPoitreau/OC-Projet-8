# src/tests/conftest.py

import sys
import threading
import faulthandler


def pytest_sessionfinish(session, exitstatus):
    print("\nThreads encore vivants à la fin de pytest:")
    for thread in threading.enumerate():
        print(
            f"- name={thread.name!r}, "
            f"daemon={thread.daemon}, "
            f"alive={thread.is_alive()}"
        )

    print("\nStack traces:")
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)