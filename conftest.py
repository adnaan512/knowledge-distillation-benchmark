"""
pytest configuration — adds the project root to sys.path so that
`import src.*` works from any test file without installation.
"""
import sys
import os

# Ensure the project root is always on the path when pytest runs
sys.path.insert(0, os.path.dirname(__file__))
