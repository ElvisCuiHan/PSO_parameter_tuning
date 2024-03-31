import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import matplotlib.pyplot as plt
import pyswarms as ps
from matplotlib.animation import FuncAnimation
import tempfile
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define Elastic_pso function
def Elastic_pso(b, X, y, cv):
    n_particles = b.shape[0]
    cost = np.zeros((n_particles, ))

    for i in range(n_particles):
        b[i, 0] = np.maximum(b[i, 0], 0.0001)
        b[i, 1] = np.minimum(np.maximum(b[i, 1], 0), 1)
        elastic_net = ElasticNet(alpha=b[i, 0], l1_ratio=b[i, 1])
        scores = cross_val_score(elastic_net, X, y, cv=cv, scoring='neg_mean_squared_error')
        cost[i] = -np.mean(scores)

    return cost

# Define Lasso_pso function
def Lasso_pso(b, X, y, cv):
    n_particles = b.shape[0]
    cost = np.zeros((n_particles, ))

    for i in range(n_particles):
        b[i] = np.maximum(b[i], 0.001)
        lasso_reg = Lasso(alpha=b[i, 0])
        scores = cross_val_score(lasso_reg, X, y, cv=cv, scoring='neg_mean_squared_error')
        cost[i] = -np.mean(scores)

    return cost

# Define Ridge_pso function
def Ridge_pso(b, X, y, cv):
    n_particles = b.shape[0]
    cost = np.zeros((n_particles, ))

    for i in range(n_particles):
        b[i] = np.maximum(b[i], 0.001)
        ridge_reg = Ridge(alpha=b[i, 0] * len(y) * 2)
        scores = cross_val_score(ridge_reg, X, y, cv=cv, scoring='neg_mean_squared_error')
        cost[i] = -np.mean(scores)

    return cost

# Define Streamlit app
def main():
    st.title("Tuning Parameter Optimization for Regularized Regression Via PSO")

    # Upload dataset
    st.subheader("Upload your dataset")
    st.markdown(
        """<span style="font-weight:bold;color:red;">The first column is the response and the rest columns are covariates.</span>""",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Number of particles and iterations in the same row
    col1, col2, col3 = st.columns(3)
    with col1:
        n_particles = st.number_input("Number of particles", min_value=5, max_value=100, step=5, value=20)
    with col2:
        n_iterations = st.number_input("Number of iterations", min_value=20, max_value=200, step=5, value=50)
    with col3:
        cv = st.number_input("Number of cross-validation folds", min_value=2, max_value=10, step=1, value=3)

    # Choose regularization type
    regularization_type = st.radio("Regularization type", ("Elastic Net", "Lasso", "Ridge"))

    # Option to show progress bar
    show_progress_bar = st.checkbox("Show progress bar", value=True)

    # Run button
    if st.button("Run"):
        if uploaded_file is not None:
            # Load dataset
            data = pd.read_csv(uploaded_file)

            # Display dataset
            st.subheader("Dataset")
            st.write(data.head(3))

            # Preprocess dataset
            scaler = StandardScaler()
            data_scale = scaler.fit_transform(data)
            y = data_scale[:, 0]
            X = data_scale[:, 1:]
            y_cache = y.copy()
            X_cache = X.copy()

            # Perform optimization using PSO
            st.text("Performing optimization using PSO...")

            progress_bar = st.progress(0)  # Initialize progress bar

            if regularization_type == "Elastic Net":
                optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.5, 'w': 0.8})
                optimize_func = Elastic_pso
            elif regularization_type == "Lasso":
                optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=1, options={'c1': 0.5, 'c2': 0.5, 'w': 0.8})
                optimize_func = Lasso_pso
            elif regularization_type == "Ridge":
                optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=1, options={'c1': 0.5, 'c2': 0.5, 'w': 0.8})
                optimize_func = Ridge_pso

            for i in range(n_iterations):
                best_cost, best_pos = optimizer.optimize(optimize_func, iters=1, X=X, y=y, cv=cv)
                progress_bar.progress((i + 1) / n_iterations)

            # Show optimization results
            st.text("Optimization finished!")
            st.text("Optimal tuning parameters:")
            result = pd.DataFrame(np.round(best_pos, 3), columns=["Tuning parameter"])
            st.write(result)
            st.text("Final positions of particles:")
            result = pd.DataFrame(np.transpose(optimizer.pos_history[-1]))
            st.write(result)

            # Plot the results
            fig, ax = plt.subplots()

            # Generate data for the function to visualize
            if regularization_type == "Elastic Net":
                # Generate data for the function to visualize
                x1 = np.linspace(0, 2, 40)
                x2 = np.linspace(0, 1, 30)
                X_mesh, Y_mesh = np.meshgrid(x1, x2)
                positions = np.stack((X_mesh.flatten(), Y_mesh.flatten()), axis=-1)
                Z = Elastic_pso(positions, X=X, y=y, cv=cv).reshape(X_mesh.shape)

                # Plot the contour plot of the loss surface
                contour = ax.contour(X_mesh, Y_mesh, Z, levels=60, cmap='Set2')
                plt.colorbar(contour, ax=ax)
                # Set labels for x and y axes
                ax.set_xlabel('Lambda')
                ax.set_ylabel('Alpha')

            else:
                if regularization_type == "Lasso":
                    x = np.linspace(0, 2, 100)  # Adjust range as needed for Lasso and Ridge
                else:
                    x = np.linspace(0, 2.5, 100)
                y = optimize_func(x.reshape(-1, 1), X=X_cache, y=y_cache, cv=cv)
                ax.set_xlabel('Tuning Parameter')
                ax.set_ylabel('Objective Function')

                # Plot the objective function
                ax.plot(x, y, label="Objective Function", color="blue")

            # Initialize scatter plot for particle positions
            particles, = ax.plot([], [], 'o', color='magenta', label='Particle Position')

            # Function to update the animation
            def update(frame):
                particle_positions = optimizer.pos_history[frame]
                if regularization_type == "Elastic Net":
                    particles.set_data(particle_positions[:, 0], particle_positions[:, 1])
                    ax.set_title(f'Iteration {frame + 1}')
                else:
                    particles.set_data(particle_positions[:, 0], optimize_func(particle_positions, X=X_cache, y=y_cache, cv=cv))
                    ax.set_title(f'Iteration {frame + 1}')

                return particles,

            # Create animation
            ani = FuncAnimation(fig, update, frames=len(optimizer.pos_history), interval=200, blit=True)

            # Save animation to temporary file
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
                ani.save(tmpfile.name, writer='pillow')

            # Display animation
            st.image(tmpfile.name)

if __name__ == "__main__":
    main()
