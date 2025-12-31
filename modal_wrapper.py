#!/usr/bin/env python3
"""
Wrapper to run modal CLI with HTTP proxy support.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply grpclib patch before anything else
import grpc_proxy_patch

# Now run modal CLI
from modal.__main__ import main

if __name__ == '__main__':
    main()
