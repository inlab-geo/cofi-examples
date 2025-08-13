#!/usr/bin/env python3
"""
Slime Mould Algorithm (SMA) Example Script

This script demonstrates the use of the Slime Mould Algorithm (SMA) for optimization
tasks using the CoFI framework. SMA is a bio-inspired metaheuristic optimizer that
mimics the oscillation mode of slime mould in nature.

The example shows:
1. Basic SMA optimization on a simple quadratic function
2. Comparison with Border Collie optimization
3. SMA variants (OriginalSMA and DevSMA)
4. Optimization on the classic Himmelblau function
"""

import numpy as np
import matplotlib.pyplot as plt
from cofi import BaseProblem, InversionOptions, Inversion


def modified_himmelblau(x):
    """Modified Himmelblau function with additional regularization term"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 + ((x[0] - 3)**2 + (x[1] - 2)**2)


def main():
    print("=" * 60)
    print("Slime Mould Algorithm (SMA) Optimization Example")
    print("=" * 60)
    
    # Example 1: Simple quadratic function
    print("\n1. Basic SMA Optimization (Quadratic Function)")
    print("-" * 50)
    
    problem1 = BaseProblem()
    problem1.set_objective(lambda x: np.sum(x**2))
    problem1.set_model_shape((5,))
    problem1.set_bounds((-10, 10))
    
    options1 = InversionOptions()
    options1.set_tool("mealpy.sma")
    options1.set_params(epoch=30, pop_size=20, seed=42)
    
    inv1 = Inversion(problem1, options1)
    result1 = inv1.run()
    
    print(f"Optimal solution: {result1.model}")
    print(f"Objective value: {result1.objective:.2e}")
    print(f"Distance from true optimum: {np.linalg.norm(result1.model):.2e}")
    
    # Example 2: Himmelblau function with SMA variants
    print("\n2. SMA Variants on Modified Himmelblau Function")
    print("-" * 50)
    
    problem2 = BaseProblem()
    problem2.set_objective(modified_himmelblau)
    problem2.set_model_shape((2,))
    problem2.set_bounds((-6, 6))
    
    for algo in ["OriginalSMA", "DevSMA"]:
        options2 = InversionOptions()
        options2.set_tool("mealpy.sma")
        options2.set_params(algorithm=algo, epoch=50, pop_size=30, seed=42)
        
        inv2 = Inversion(problem2, options2)
        result2 = inv2.run()
        
        distance = np.linalg.norm(result2.model - [3, 2])
        print(f"{algo:12} | Solution: [{result2.model[0]:6.3f}, {result2.model[1]:6.3f}] | "
              f"Objective: {result2.objective:8.3f} | Distance: {distance:.3f}")
    
    # Example 3: Comparison with Border Collie optimization
    print("\n3. SMA vs Border Collie Optimization")
    print("-" * 50)
    
    methods = [
        ("mealpy.sma", {"algorithm": "OriginalSMA", "epoch": 50, "pop_size": 30}),
        ("cofi.border_collie_optimization", {"number_of_iterations": 50})
    ]
    
    for method_name, params in methods:
        try:
            options3 = InversionOptions()
            options3.set_tool(method_name)
            options3.set_params(seed=42, **params)
            
            inv3 = Inversion(problem2, options3)
            result3 = inv3.run()
            
            distance = np.linalg.norm(result3.model - [3, 2])
            print(f"{method_name:25} | Solution: [{result3.model[0]:6.3f}, {result3.model[1]:6.3f}] | "
                  f"Distance: {distance:.3f}")
        except Exception as e:
            print(f"{method_name:25} | Error: {str(e)[:40]}")
    
    # Example 4: Demonstrate CoFI's "define once, solve many ways" principle
    print("\n4. CoFI Architecture Demonstration")
    print("-" * 50)
    print("Same problem definition, multiple optimizers:")
    
    solvers = [
        ("mealpy.sma", {"algorithm": "OriginalSMA", "epoch": 30, "pop_size": 20}),
        ("mealpy.slime_mould", {"algorithm": "DevSMA", "epoch": 30, "pop_size": 20}),  # Test alias
        ("scipy.optimize.minimize", {"method": "Powell"})
    ]
    
    for solver_name, solver_params in solvers:
        try:
            options4 = InversionOptions()
            options4.set_tool(solver_name)
            options4.set_params(seed=42, **solver_params)
            
            inv4 = Inversion(problem2, options4)  # Same problem definition!
            result4 = inv4.run()
            
            distance = np.linalg.norm(result4.model - [3, 2])
            print(f"{solver_name:25} | Solution: [{result4.model[0]:6.3f}, {result4.model[1]:6.3f}] | "
                  f"Distance: {distance:.3f}")
        except Exception as e:
            print(f"{solver_name:25} | Error: {str(e)[:40]}")
    
    print("\n" + "=" * 60)
    print("SMA Integration Demo Complete!")
    print("CoFI's Slime Mould Algorithm tool successfully integrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()