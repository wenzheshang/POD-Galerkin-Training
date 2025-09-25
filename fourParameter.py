import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os

# ---------- Set global font ----------
mpl.rcParams['font.family'] = 'Times New Roman'

# ---------- PARAMETERS ----------
velocity_vals = [0.03, 0.04, 0.05, 0.06, 0.07]   # 5
temperature_vals = [290, 292, 294, 296, 298]    # 5
angle1_vals = [0, 15]                          # 2
angle2_vals = [15, 30, 45, 60, 75]              # 5

block_cols = len(angle2_vals)   # 5
block_rows = len(angle1_vals)   # 2
n_vel = len(velocity_vals)      # 5
n_temp = len(temperature_vals)  # 5

big_cols = n_vel * block_cols   # 25
big_rows = n_temp * block_rows  # 10

# ---------- Read result.csv ----------
csv_path = "result.csv"
if os.path.exists(csv_path):
    df_exist = pd.read_csv(csv_path)
    df_exist.columns = [c.strip().capitalize() for c in df_exist.columns]
else:
    print("⚠️ result.csv not found, using demo empty set")
    df_exist = pd.DataFrame(columns=["Velocity","Temperature","Angle1","Angle2"])

# build set of existing tuples
existing_set = set()
for _, r in df_exist.iterrows():
    try:
        existing_set.add((round(float(r["Velocity"]), 3),
                          int(r["Temperature"]),
                          int(r["Angle1"]),
                          int(r["Angle2"])))
    except Exception:
        pass

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(12, 8))

# draw cells
for vi, v in enumerate(velocity_vals):
    for ti, t in enumerate(temperature_vals):
        base_x = vi * block_cols
        base_y = ti * block_rows
        for a1_idx, a1 in enumerate(angle1_vals):
            for a2_idx, a2 in enumerate(angle2_vals):
                x = base_x + a2_idx
                y = base_y + a1_idx
                combo = (round(v, 3), t, a1, a2)
                if combo in existing_set:
                    color = "#FFBE7A"   # existing = fill color
                else:
                    color = "white"
                rect = patches.Rectangle((x, y), 1, 1,
                                         facecolor=color,
                                         edgecolor="gray",
                                         linewidth=0.3)
                ax.add_patch(rect)

                # 如果是存在的组合，就在小格子里标注角度组合
                if combo in existing_set:
                    ax.text(x + 0.5, y + 0.5, f"({a2},{a1})",
                            ha="center", va="center", fontsize=10, color="black")

        # draw big block boundary (red frame with custom color)
        big_rect = patches.Rectangle((base_x, base_y), block_cols, block_rows,
                                     fill=False, edgecolor="#FA7F6F", linewidth=1.2)
        ax.add_patch(big_rect)

# 设置轴标签
x_ticks = [vi * block_cols + block_cols/2 for vi in range(n_vel)]
y_ticks = [ti * block_rows + block_rows/2 for ti in range(n_temp)]

ax.set_xticks(x_ticks)
ax.set_xticklabels([str(v) for v in velocity_vals], fontsize=11)
ax.set_yticks(y_ticks)
ax.set_yticklabels([str(t) for t in temperature_vals], fontsize=11)

ax.set_xlabel("Massflow (kg/s)", fontsize=13)
ax.set_ylabel("Temperature (K)", fontsize=13)
ax.set_title("25x10 Grid: Velocity x Temperature blocks\nEach block = 5x2 (Angle2 x Angle1)", fontsize=14)

ax.set_xlim(0, big_cols)
ax.set_ylim(0, big_rows)
ax.set_aspect("equal")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("fourparameter.svg",dpi=900)
plt.show()
