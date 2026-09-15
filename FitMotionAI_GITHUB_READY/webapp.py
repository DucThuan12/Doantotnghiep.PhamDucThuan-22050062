"""Backward-compatible entry point.

The old prototype duplicated routes and used a second, inconsistent rep-counting
algorithm. It now exposes the canonical Flask application from ``app.py`` so
there is only one production web pipeline.
"""
from app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
