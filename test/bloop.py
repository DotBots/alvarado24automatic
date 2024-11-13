import sympy as sp

# Define the variable x
x = sp.symbols('x')

# Define the polynomial expression
expr = (x + 1) * (x - 2 + 3*sp.I) * (x - 2 - 3*sp.I)

# Expand the polynomial
expanded_expr = sp.expand(expr)

# Print the expanded result
print(expanded_expr)
