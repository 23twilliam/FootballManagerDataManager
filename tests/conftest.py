import os
import sys
from pathlib import Path

# Headless backend, set before anything imports pyplot.
os.environ.setdefault('MPLBACKEND', 'Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
