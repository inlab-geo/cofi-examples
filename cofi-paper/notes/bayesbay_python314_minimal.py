#!/usr/bin/env python3
"""
Minimal reproducer for BayesBay Python 3.14 compatibility issue.

Run with: python bayesbay_python314_minimal.py

On Python 3.14+, this will fail with TypeError.
On Python 3.13 and earlier, this works fine.
"""
import sys
print(f"Python version: {sys.version}")

import numpy as np
print(f"NumPy version: {np.__version__}")

import bayesbay
print(f"BayesBay version: {bayesbay.__version__}")

print("\n" + "="*60)
print("Test 1: Direct call to nearest_neighbour_1d with numpy scalars")
print("="*60)

from bayesbay._utils_1d import nearest_neighbour_1d

# Create test data
x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
query_point = np.float64(4.0)  # numpy scalar, not Python float
xlen = np.intp(5)              # numpy scalar, not Python int

print(f"x = {x}")
print(f"query_point = {query_point} (type: {type(query_point).__name__})")
print(f"xlen = {xlen} (type: {type(xlen).__name__})")

try:
    result = nearest_neighbour_1d(xp=query_point, x=x, xlen=xlen)
    print(f"SUCCESS: nearest_neighbour_1d returned {result}")
except TypeError as e:
    print(f"FAILED: {e}")

print("\n" + "="*60)
print("Test 2: Using Voronoi1D (triggers the issue internally)")
print("="*60)

from bayesbay.discretization import Voronoi1D

voronoi = Voronoi1D(
    name="test",
    vmin=0.0,
    vmax=10.0,
    perturb_std=0.5,
)

try:
    # Initialize creates internal state
    state = voronoi.initialize()
    print(f"initialize() SUCCESS: {state}")

    # birth() calls nearest_neighbour internally with numpy scalars
    new_state, log_prob = voronoi.birth(state)
    print(f"birth() SUCCESS: {new_state}")
except TypeError as e:
    print(f"FAILED: {e}")

print("\n" + "="*60)
print("Test 3: Workaround - convert to Python scalars")
print("="*60)

try:
    # This works on all Python versions
    result = nearest_neighbour_1d(
        xp=float(query_point),  # explicit conversion
        x=x,
        xlen=int(xlen)          # explicit conversion
    )
    print(f"SUCCESS with explicit conversion: {result}")
except TypeError as e:
    print(f"FAILED even with conversion: {e}")
