import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image
from ase.io import read
from nequip.ase import NequIPCalculator
import torch

structures = read("nequipreference.xyz", index=":")

calculator = NequIPCalculator.from_compiled_model(
    compile_path="bestcompiledfosforene.nequip.pt2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
   
    atoms = structures[frame]
    forces_true = atoms.get_forces()
    positions = atoms.get_positions()

    atoms.calc = calculator
    forces_pred = atoms.get_forces()

    positions_2d = positions[:, :2]
    forces_true_2d = forces_true[:, :2]
    forces_pred_2d = forces_pred[:, :2]

    ax.set_title(f"Estructura {frame+1}/{len(structures)}", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(positions_2d[:, 0].min() - 1, positions_2d[:, 0].max() + 1)
    ax.set_ylim(positions_2d[:, 1].min() - 1, positions_2d[:, 1].max() + 1)

    ax.quiver(positions_2d[:, 0], positions_2d[:, 1],
              forces_pred_2d[:, 0], forces_pred_2d[:, 1],
              color="black", scale=1, scale_units="xy", angles="xy", label="Predicted")

    ax.quiver(positions_2d[:, 0], positions_2d[:, 1],
              forces_true_2d[:, 0], forces_true_2d[:, 1],
              color="red", scale=1, scale_units="xy", angles="xy", label="True")

    ax.scatter(positions_2d[:, 0], positions_2d[:, 1], color="black", s=10, label="Positions")
    ax.legend()

ani = FuncAnimation(
    fig, update,
    frames=len(structures),
    interval=500,
    blit=False,
    save_count=len(structures)
)

gif_path = "forces_comparasion.gif"
ani.save(gif_path, writer=PillowWriter(fps=2))
plt.close()

display(Image(filename=gif_path))