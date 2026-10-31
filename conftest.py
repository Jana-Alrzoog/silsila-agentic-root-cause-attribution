import os
import sys

# Ensure silsila_v2 is always at the top of sys.path, taking precedence over editable installs
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
