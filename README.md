# 🏀 MarchMadSim: March Madness Simulator

MarchMadSim is a Python-based Monte Carlo simulation for predicting outcomes in the NCAA March Madness tournament. Using advanced statistical modeling, team adjustments, and probabilistic methods, it runs thousands of tournament simulations to determine the most probable outcomes. 

## 📌 Features

- 🏆 **Monte Carlo Tournament Simulations** – Simulates thousands of brackets to determine the most likely winners.
- 📊 **Advanced Win Probability Model** – Accounts for efficiency, momentum, fatigue, tempo, and randomness.
- 🔥 **Fatigue Modeling** – Teams get more fatigued in later rounds and in closer games.
- 🚀 **First Round Matchups Generation** – Automatically generates matchups based on real tournament seeds.
- 📈 **Detailed Stats Tracking** – Tracks win percentages, championship likelihoods, and deep run probabilities.
- 🔄 **Customizable Parameters** – Adjust weight factors for momentum, fatigue, and playstyle influence.
- 🎯 **High-Performance Simulations** – Optimized for running thousands of tournament iterations efficiently.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MarchMadSim.git
   cd MarchMadSim
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation:
   ```bash
   python simulate.py
   ```

## 🛠 How It Works

### 🎲 Win Probability Calculation
Each game is simulated using an advanced probability function based on:
- **Team Efficiency** (adjusted Net Rating)
- **Random Shocks** (heavy-tailed noise for upsets)
- **Momentum & Luck** (computed from season stats)
- **Playstyle Differences** (adjusted tempo comparison)
- **Fatigue Effects** (progressively impacts teams after close games)

The probability is computed using:
```python
p = 1 / (1 + np.exp(-effective_diff / effective_sigma))
```
where `effective_diff` is the weighted sum of all these factors.

### 📊 Simulating the Bracket
The tournament is simulated in two modes:
1. **Single Tournament Run** – Runs one tournament and tracks winners.
2. **Monte Carlo Mode** – Runs thousands of brackets and aggregates results.

Each round follows these steps:
1. Compute **win probability** for each game.
2. Simulate **N** games per matchup.
3. **Advance most probable winners** to the next round.

##  Example Output
For the mode where every bracket is fully simulated and a ranking of probable winners is generated (example output not a real simulation but just illustrative):

```plaintext

Champion Win Counts (Top 8): 
Maryland: 224 wins (22.40%)
Texas Tech: 218 wins (21.80%)
Colorado St.: 217 wins (21.70%)
...
```
For the Simulations where we simulated each game multiple times to find the most probable winner per game (example output not a real simulation but just illustrative):

```plaintext

Tournament Outcome (mode-based round-by-round simulation):
-----------------------------------------------------------

Round of 64 Complete:
Florida vs Norfolk St. -> Norfolk St. (696/1000 outcomes, 69.60%)
Connecticut vs Oklahoma -> Oklahoma (725/1000 outcomes, 72.50%)
Memphis vs Colorado St. -> Colorado St. (861/1000 outcomes, 86.10%)
...

Round of 32 Complete:
Norfolk St.  vs Oklahoma -> Oklahoma (777/1000 outcomes, 67.70%)
Colorado St. vs Maryland -> Maryland (751/1000 outcomes, 75.10%)
...

Sweet 16 Complete:
Oklahoma vs Maryland -> Oklahoma (630/1000 outcomes, 63.00%)
Duke vs St. John's -> Duke (856/1000 outcomes, 85.60%)
...

Elite 8 Complete:
Oklahoma vs Duke Tech -> Duke (788/1000 outcomes, 78.80%)
Auburn vs Michigan St. -> Auburn (773/1000 outcomes, 77.30%)
...

Championship:
Duke vs Auburn -> Duke (514/1000 outcomes, 51.40%)

Champion: Duke
```
## ⚙️ Customization
Modify parameters in the params data structure
```python
params = {
    "shock_scale": 0.1,   # Randomness factor
    "luck_factor": 0.1,   # Impact of luck
    "momentum_factor": 0.2, # Impact of momentum
    "style_factor": 0.2,  # Impact of playstyle
    "fatigue_factor": 0.05 # Fatigue increase per close game
}
```


## 🤝 Contributing
Want to improve MarchMadSim? Open a pull request or submit an issue!

## 📜 License
This project is licensed under the MIT License.

---

⭐ If you like this project, give it a star on GitHub! ⭐
