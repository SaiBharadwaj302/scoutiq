import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from api.main import app
import uvicorn

if __name__ == "__main__":
    # 0.0.0.0, not 127.0.0.1: this process runs inside a Docker container in
    # both docker-compose and the deployed EC2 setup. Binding to loopback
    # would make it unreachable through Docker's port publishing entirely —
    # it would only ever answer requests from inside its own network
    # namespace, never from the host or the outside world.
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104