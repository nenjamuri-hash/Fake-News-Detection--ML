import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path("fog_diagrams")
OUTPUT_DIR.mkdir(exist_ok=True)

def box(ax, x, y, text, w=3, h=1):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3",
        linewidth=2,
        edgecolor="black",
        facecolor="#f0dab1"
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=12)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", lw=2))

def save(fig, name):
    out_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# -----------------------------------------------------
# 1. System Architecture
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.axis("off")

box(ax, 4, 5, "Cloud Server")
box(ax, 1, 3, "Fog Node 1")
box(ax, 4, 3, "Fog Node 2")
box(ax, 7, 3, "Fog Node 3")

box(ax, 0.5, 1, "User 1", 2,1)
box(ax, 3.5, 1, "User 2", 2,1)
box(ax, 6.5, 1, "User 3", 2,1)
box(ax, 9.5, 1, "User 4", 2,1)

arrow(ax, 1.5, 2, 2.5, 3)
arrow(ax, 4.5, 2, 5, 3)
arrow(ax, 7.5, 2, 7.5, 3)
arrow(ax, 10.5, 2, 9, 3)

arrow(ax, 2.5, 4, 5, 5)
arrow(ax, 5.5, 4, 5, 5)
arrow(ax, 8.5, 4, 5, 5)

save(fig, "architecture")

# -----------------------------------------------------
# 2. Markov Predictor Diagram
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6,10))
ax.axis("off")

steps = [
    "Request Sequence",
    "Learn Transitions",
    "Transition Matrix",
    "Predict Next Item",
    "Prefetch Into Cache"
]

y = 8
for step in steps:
    box(ax, 2, y, step, 6, 1)
    y -= 2

y = 7.5
for _ in steps[:-1]:
    arrow(ax, 5, y, 5, y-1.5)
    y -= 2

save(fig, "predictor")

# -----------------------------------------------------
# 3. Request Flow Diagram
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.axis("off")

box(ax, 1, 4, "User Request")
box(ax, 5, 4, "Fog Node")

arrow(ax, 3, 4.5, 5, 4.5)

box(ax, 5, 2.5, "Cache Hit?")
arrow(ax, 6.5, 4, 6.5, 3)

box(ax, 1, 1, "Serve From Cache")
arrow(ax, 6.5, 2.5, 2.5, 2)

box(ax, 9, 1, "Fetch From Cloud")
arrow(ax, 6.5, 2.5, 9.5, 2)

save(fig, "request_flow")

# -----------------------------------------------------
# 4. Simple Fog Workflow
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.axis("off")

box(ax, 1, 4, "User")
box(ax, 5, 4, "Fog Node")
arrow(ax, 3,4.5,5,4.5)

box(ax, 5, 2.5, "Cache Hit?")
arrow(ax, 6.5,4,6.5,3)

box(ax, 1,1, "Serve Local")
arrow(ax, 6.5,2.5,2.5,2)

box(ax, 9,1, "Cloud Server")
arrow(ax, 6.5,2.5,9.5,2)

save(fig, "simple_fog")

# -----------------------------------------------------
# 5. Cooperative Fog Workflow
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10,7))
ax.axis("off")

box(ax, 1,5,"User")
box(ax, 5,5,"Fog Node")
arrow(ax, 3,5.5,5,5.5)

box(ax,5,3.5,"Local Hit?")
arrow(ax,6.5,5,6.5,4)

box(ax,2,2,"Serve Local")
arrow(ax,6.5,3.5,3.5,2.5)

box(ax,8,3.5,"Neighbor Hit?")
arrow(ax,6.5,3.5,8,4)

box(ax,8,2,"Serve From Neighbor")
arrow(ax,9,3.5,9,2.5)

box(ax,5,1,"Cloud")
arrow(ax,6.5,3.5,6,1.5)

save(fig, "cooperative_fog")

# -----------------------------------------------------
# 6. Predictive Fog Workflow
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10,10))
ax.axis("off")

box(ax,1,7,"User")
box(ax,5,7,"Fog Node QoS")
arrow(ax,3,7.5,5,7.5)

box(ax,5,5.5,"Cache Hit?")
arrow(ax,6.5,7,6.5,6)

box(ax,1,4,"Serve Local")
arrow(ax,6.5,5.5,2.5,5)

box(ax,8,5.5,"Neighbor Hit?")
arrow(ax,6.5,5.5,8,6)

box(ax,8,4,"Serve Neighbor")
arrow(ax,9,5.5,9,4.5)

box(ax,5,3,"Cloud")
arrow(ax,6.5,5.5,5.5,3.5)

box(ax,5,1,"Markov Predictor")
arrow(ax,5.5,3,5.5,1.5)
arrow(ax,5.5,1,5.5,7)

save(fig, "predictive_fog")

# -----------------------------------------------------
# 7. QoS Cache Flow
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8,10))
ax.axis("off")

box(ax,3,8,"New Item")
box(ax,3,6.5,"Cache Full?")
arrow(ax,4.5,8.5,4.5,7)

box(ax,0.5,5,"Insert Item")
arrow(ax,4.5,6.5,2,5.5)

box(ax,5.5,5.5,"Non-QoS Exists?")
arrow(ax,4.5,6.5,6,6)

box(ax,5.5,4,"Evict Non-QoS")
arrow(ax,6,5.5,6,4.5)

box(ax,2,4,"Evict LRU")
arrow(ax,4.5,6.5,2,4.5)

arrow(ax,2,4,2,5)
arrow(ax,6,4,2,5)

save(fig, "qos_cache_flow")

print("\nAll diagrams generated successfully!")
