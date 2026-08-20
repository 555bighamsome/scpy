# IICCSSS 2026 - Hackathon: RL Masterclass, Traditional vs Recurrent Modeling of Behavior

This is the repository for the IICCSSS 2026 Hackathon on Traditional vs Recurrent Modeling of Behavior. It extends Sarah Master's work on the Azulejos dataset, which is a collection of behavioral data from bandit tasks. The goal of this hack is to explore and compare classical reinforcement-learning models with recurrent neural networks in modeling learning behavior, and to explore the data to answer self-defined research questions. 

Setup Recommendations:

1. Clone the repository:
```bash
git clone [https://github.com/maltekrambeer/rl_behavioral_modeling.git](https://github.com/maltekrambeer/rl_behavioral_modeling.git)
cd rl_behavioral_modeling
```

2. Create a virtual environment and activate it:
```bash
# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Hackathon study: Bad Luck or a Changing World?

Our focused study asks how people distinguish an isolated bad outcome from a genuine change in reward probability.

- [Read the full English research document](BAD_LUCK_STUDY.md)
- [Open the public visual research brief](https://555bighamsome.github.io/scpy/bad-luck.html)
- [Inspect and reproduce the analysis](analysis/README.md)
- [View the compact result tables](results/bad_luck_study/)
