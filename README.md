# QDax with Evosax integration

## Install
The Kheperax version used here is a custom fork.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py algo=jedi task=kh_standard
```  

## Features roadmap

### Algorithms

- JEDi:
- [X] JEDi emitter
- [ ] Weighted GP
- [ ] Decay alpha

- ES:
- [ ] Standalone ES emitter
- [ ] NS-ES

- MEES:
- [ ] MEES emitter
- [ ] MEMES emitter

- CMAME:
- [X] CMAME emitter
- [X] CMA-MAE
- [ ] CMA-MEGA

### Tasks

- [X] Brax
- Kheperax
    - [X] Custom fork
    - [X] Kheperax 0.2.0

### Pipeline

- [X] Hydra
- [X] W&B

