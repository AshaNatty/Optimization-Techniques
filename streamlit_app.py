import streamlit as st
import sympy as sp

def newton_minima_multivariable(f_expr, vars, x0, tol):
    gradients = [sp.diff(f_expr, var) for var in vars]  # Gradient (first derivatives)
    hessian = sp.Matrix([[sp.diff(g, var) for var in vars] for g in gradients])  # Hessian matrix
    
    iterations = []  # Store iteration details
    xn = sp.Matrix(x0)
    
    st.write(f"Function: f({', '.join([str(v) for v in vars])}) = {f_expr}")
    st.write("1️⃣ The gradient (∇f) represents the first derivatives with respect to the variables:")
    st.write(f"∇f = [{', '.join([f'∂f/∂{var}' for var in vars])}]")
    st.write(f"Gradient Formula: {sp.Matrix(gradients)}")
    
    st.write("2️⃣ Hessian Matrix:")
    st.write("The Hessian matrix (H) represents the second derivatives of f(x,y):")
    st.latex(r"""
    H = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
    \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
    \end{bmatrix}
    """)
    st.write(f"Hessian Matrix Formula: H = {hessian}")
    
    step = 0
    while True:
        f_x = round(f_expr.subs(zip(vars, xn)).evalf(), 4)
        grad_values = sp.Matrix([round(g.subs(zip(vars, xn)).evalf(), 4) for g in gradients])
        hessian_values = sp.Matrix([[round(h.subs(zip(vars, xn)).evalf(), 4) for h in row] for row in hessian.tolist()])
        
        st.write(f"Step {step}:")
        st.write(f"  1. Current x = {xn.applyfunc(lambda v: round(v, 4))}")
        st.write(f"  2. Function value f(x) = {f_x}")
        st.write(f"  3. Gradient = {grad_values}")
        st.write(f"  4. Hessian Matrix = {hessian_values}")
        
        iterations.append((step, xn, f_x, grad_values, hessian_values))
        
        if grad_values.norm() < tol:
            st.write("  5. Gradient norm is below tolerance, stopping.")
            break  # Stop if gradient is small enough
        
        if hessian_values.det() == 0:
            st.error("Hessian matrix is singular, Newton's method fails.")
            st.write("  6. Hessian matrix is singular, method fails.")
            return []
        
        st.write("  6. Formula: x_new = x - H^(-1) * Gradient")
        xn_new = xn - hessian_values.inv() * grad_values
        st.write(f"  7. Next x = {xn_new.applyfunc(lambda v: round(v, 4))}")
        
        if (xn_new - xn).norm() < tol:
            xn = xn_new
            break  # Stop if change is small enough
        
        xn = xn_new
        step += 1
    
    return iterations, xn

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
            for step, xn, f_x, grad_values, hessian_values in steps:
                st.write(f"Step {step}:")
                st.write(f"  1. x = {xn.applyfunc(lambda v: round(v, 4))}")
                st.write(f"  2. f(x) = {f_x}")
                st.write(f"  3. Gradient = {grad_values}")
                st.write(f"  4. Hessian = {hessian_values}")
                st.write("  5. Formula: x_new = x - H^(-1) * Gradient")
                st.write("-----------------------------------")
            
            st.success(f"Estimated Minima at x = {minima.applyfunc(lambda v: round(v, 4))}")
    except Exception as e:
        st.error(f"Error: {e}")
