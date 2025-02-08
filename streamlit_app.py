import streamlit as st
import sympy as sp

def newton_minima_multivariable(f_expr, vars, x0, tol):
    gradients = [sp.diff(f_expr, var) for var in vars]  # Gradient (first derivatives)
    hessian = sp.Matrix([[sp.diff(g, var) for var in vars] for g in gradients])  # Hessian matrix
    
    iterations = []  # Store iteration details
    xn = sp.Matrix(x0)
    
    while True:
        grad_values = sp.Matrix([g.subs(zip(vars, xn)).evalf() for g in gradients])
        hessian_values = hessian.subs(zip(vars, xn)).evalf()
        
        if grad_values.norm() < tol:
            break  # Stop if gradient is small enough
        
        if hessian_values.det() == 0:
            st.error("Hessian matrix is singular, Newton's method fails.")
            return []
        
        xn_new = xn - hessian_values.inv() * grad_values
        
        iterations.append((xn, grad_values, hessian_values, xn_new))
        
        if (xn_new - xn).norm() < tol:
            break  # Stop if change is small enough
        
        xn = xn_new
    
    return iterations, xn_new

# Streamlit UI
st.title("Newton's Method for Multivariable Minima")

# User Inputs
equation = st.text_input("Enter function f(x, y):", "x**2 + y**2 - 4*x*y + x")
vars_input = st.text_input("Enter variables (comma-separated):", "x,y")
initial_guess = st.text_input("Enter initial guess (comma-separated):", "1,1")
tol = st.number_input("Enter stopping condition (tolerance):", value=0.0001, format="%.5f")

if st.button("Find Minima"):
    try:
        vars = [sp.symbols(var.strip()) for var in vars_input.split(',')]
        f_expr = sp.sympify(equation)  # Convert string to sympy expression
        x0 = [float(num.strip()) for num in initial_guess.split(',')]
        
        steps, minima = newton_minima_multivariable(f_expr, vars, x0, tol)
        
        if steps:
            st.subheader("Iteration Steps")
            for i, (xn, grad_values, hessian_values, xn_new) in enumerate(steps):
                st.write(f"Step {i+1}: x = {xn}, Gradient = {grad_values}, Hessian = {hessian_values}, Next x = {xn_new}")
            
            st.success(f"Estimated Minima at x = {minima}")
    except Exception as e:
        st.error(f"Error: {e}")
