import csv
import matplotlib.pyplot as plt

COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
ROLE_NAMES = ['Red1', 'Red2', 'Yellow1', 'Yellow2']

LOG_PATH = "ball_control_log.csv"
NUM_PLAYERS = 4
player_control_counts = [0] * NUM_PLAYERS

with open(LOG_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        _, _, idx, _, _ = row
        idx = int(idx)
        if idx != -1:
            player_control_counts[idx] += 1

total_control = sum(player_control_counts)
if total_control == 0:
    print("No ball control detected.")
    exit()

labels = [ROLE_NAMES[i] for i in range(NUM_PLAYERS) if player_control_counts[i] > 0]
sizes = [player_control_counts[i] for i in range(NUM_PLAYERS) if player_control_counts[i] > 0]
colors = [COLORS[i] for i in range(NUM_PLAYERS) if player_control_counts[i] > 0]

plt.figure(figsize=(6, 6))
patches, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12}
)

for text in texts:
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(11)

plt.title("Ball Control Distribution (Controlled Steps Only)", fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig("ball_control_pie.png", dpi=300)
plt.show()
