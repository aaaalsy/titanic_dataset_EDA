"""
Task 02: Data Cleaning & Exploratory Data Analysis (EDA) — Titanic Dataset
Dataset: Titanic (built-in via seaborn or CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 02 — Titanic EDA")
print("=" * 60)

try:
    df = sns.load_dataset('titanic')
    print("Loaded Titanic dataset from seaborn.")
except Exception:
    # Fallback: create a representative mock dataset
    np.random.seed(0)
    n = 891
    df = pd.DataFrame({
        'survived':  np.random.choice([0, 1], n, p=[0.62, 0.38]),
        'pclass':    np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'sex':       np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'age':       np.where(np.random.rand(n) < 0.2, np.nan,
                              np.clip(np.random.normal(30, 14, n), 1, 80)),
        'sibsp':     np.random.choice(range(9), n),
        'parch':     np.random.choice(range(7), n),
        'fare':      np.clip(np.random.exponential(33, n), 0, 512),
        'embarked':  np.random.choice(['S', 'C', 'Q', np.nan], n, p=[0.72, 0.19, 0.086, 0.004]),
        'class':     np.random.choice(['First', 'Second', 'Third'], n),
        'who':       np.random.choice(['man', 'woman', 'child'], n),
        'alone':     np.random.choice([True, False], n),
    })
    print("Using simulated Titanic-like dataset.")

# ── 2. Data Cleaning ─────────────────────────────────────────────────────────
print("\n── Raw Shape:", df.shape)
print("── Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
if 'deck' in df.columns:
    df.drop(columns=['deck'], inplace=True)
if 'embark_town' in df.columns:
    df.drop(columns=['embark_town'], inplace=True)
if 'alive' in df.columns:
    df.drop(columns=['alive'], inplace=True)

df.drop_duplicates(inplace=True)
print("\n── Cleaned Shape:", df.shape)
print("── Null count after cleaning:", df.isnull().sum().sum())

# Save cleaned CSV
df.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved → titanic_cleaned.csv")

# ── 3. Summary Statistics ─────────────────────────────────────────────────────
print("\n── Descriptive Statistics:")
print(df[['age', 'fare', 'survived']].describe().round(2))

survival_rate = df['survived'].mean() * 100
print(f"\n── Overall Survival Rate: {survival_rate:.1f}%")
print("── Survival by Gender:\n", df.groupby('sex')['survived'].mean().mul(100).round(1))
print("── Survival by Class:\n",  df.groupby('pclass')['survived'].mean().mul(100).round(1))

# ── 4. Visualisations ────────────────────────────────────────────────────────
BG = '#1e1e2e'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'text.color': 'white', 'axes.labelcolor': 'white',
                     'xtick.color': 'white', 'ytick.color': 'white',
                     'axes.edgecolor': '#444'})

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 4a. Survival count
ax1 = fig.add_subplot(gs[0, 0])
surv_counts = df['survived'].value_counts()
ax1.bar(['Died', 'Survived'], surv_counts.values,
        color=['#ef5350', '#66bb6a'], edgecolor='white', linewidth=0.4)
ax1.set_title('Survival Count', fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(surv_counts.values):
    ax1.text(i, v + 5, str(v), ha='center', color='white', fontsize=10)
ax1.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)

# 4b. Survival by Gender
ax2 = fig.add_subplot(gs[0, 1])
gender_surv = df.groupby('sex')['survived'].mean().mul(100)
ax2.bar(gender_surv.index, gender_surv.values,
        color=['#4fc3f7', '#f48fb1'], edgecolor='white', linewidth=0.4)
ax2.set_title('Survival Rate by Gender (%)', fontweight='bold')
ax2.set_ylabel('Survival Rate (%)')
for i, (g, v) in enumerate(gender_surv.items()):
    ax2.text(i, v + 1, f'{v:.1f}%', ha='center', color='white', fontsize=10)
ax2.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)

# 4c. Survival by Pclass
ax3 = fig.add_subplot(gs[0, 2])
class_surv = df.groupby('pclass')['survived'].mean().mul(100)
ax3.bar([f'Class {c}' for c in class_surv.index], class_surv.values,
        color=['#ffd54f', '#4db6ac', '#ef9a9a'], edgecolor='white', linewidth=0.4)
ax3.set_title('Survival Rate by Passenger Class (%)', fontweight='bold')
ax3.set_ylabel('Survival Rate (%)')
for i, v in enumerate(class_surv.values):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', color='white', fontsize=10)
ax3.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax3.set_axisbelow(True)

# 4d. Age distribution
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.hist(df[df['survived'] == 0]['age'], bins=30, alpha=0.7, color='#ef5350', label='Died', edgecolor='white', linewidth=0.3)
ax4.hist(df[df['survived'] == 1]['age'], bins=30, alpha=0.7, color='#66bb6a', label='Survived', edgecolor='white', linewidth=0.3)
ax4.set_title('Age Distribution by Survival', fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.legend(facecolor='#2a2a3e', labelcolor='white')
ax4.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax4.set_axisbelow(True)

# 4e. Fare distribution (log scale)
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df['fare'] + 1, bins=40, color='#ab47bc', edgecolor='white', linewidth=0.3, log=True)
ax5.set_title('Fare Distribution (log scale)', fontweight='bold')
ax5.set_xlabel('Fare')
ax5.set_ylabel('Count (log)')
ax5.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
ax5.set_axisbelow(True)

# 4f. Heatmap — correlation
ax6 = fig.add_subplot(gs[2, 0:2])
num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
available = [c for c in num_cols if c in df.columns]
corr = df[available].corr()
im = ax6.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax6.set_xticks(range(len(available)))
ax6.set_yticks(range(len(available)))
ax6.set_xticklabels(available, rotation=45, ha='right')
ax6.set_yticklabels(available)
ax6.set_title('Correlation Heatmap', fontweight='bold')
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
for i in range(len(available)):
    for j in range(len(available)):
        ax6.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                 color='white', fontsize=8)

# 4g. Embarked pie
ax7 = fig.add_subplot(gs[2, 2])
emb_counts = df['embarked'].value_counts()
ax7.pie(emb_counts.values, labels=emb_counts.index,
        colors=['#4fc3f7', '#ffd54f', '#f48fb1'],
        autopct='%1.1f%%', textprops={'color': 'white'},
        wedgeprops={'edgecolor': '#1e1e2e', 'linewidth': 1.5})
ax7.set_title('Embarkation Port Distribution', fontweight='bold')

fig.suptitle('Task 02 — Titanic EDA Dashboard', fontsize=16,
             fontweight='bold', color='white', y=1.01)

plt.savefig('task02_output.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("\nEDA dashboard saved → task02_output.png")
