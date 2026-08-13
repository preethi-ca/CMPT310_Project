import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Results from linear-regression.py
lambda_labels = ["0", "0.01", "1", "10"]

rmse_values = [0.4195, 0.4195, 0.4193, 0.4188]
mae_values = [0.3202, 0.3202, 0.3202, 0.3205]
r2_values = [0.0458, 0.0458, 0.0468, 0.0506]

# -----------------------------
# RMSE vs Lambda
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(lambda_labels, rmse_values, marker="o", linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title("RMSE vs Lambda")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rmse_vs_lambda.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# MAE vs Lambda
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(lambda_labels, mae_values, marker="o", linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("MAE")
plt.title("MAE vs Lambda")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mae_vs_lambda.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# R² vs Lambda
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(lambda_labels, r2_values, marker="o", linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("R²")
plt.title("R² vs Lambda")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "r2_vs_lambda.png", dpi=300, bbox_inches="tight")
plt.show()
