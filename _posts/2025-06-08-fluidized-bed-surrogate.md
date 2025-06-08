---
layout: post
title: "Surrogate Modeling for Pressure Drop using PartiNet"
date: 2025-06-07
categories: [projects, notebooks]
tags: [cfd, surrogate-models, ml, fluidized-bed]
---

# PartiNet v1: Surrogate Modeling for Fluidized Beds

This script uses synthetic data from theoretical models to predict pressure drop
in gas-solid fluidized beds using scikit-learn models.

### Load the libraries


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
```

## Variable Used in the Dataset

$U_g$ : Superficial Gas Velocity, randomly generated between 0.1 m/s and 2.0 m/s

$d_p$ : Particle Diameter, randomly generated between $50~\mu m$ and $500~\mu m$

$\rho_p$ : Particle Density, randomly generated between $1000~kg/m^3$ and $3000~kg/m^3$

$H$ : Bed Height, randomly generated between $0.2~m$ and $1.0~m$

## Target Variable

$dP$ : Pressure Drop across the fluidized bed, which is a function of the superficial gas velocity, bed height, particle properties and fluidization state.


```python
# Generate synthetic dataset
np.random.seed(42)
n = 200
Ug = np.random.uniform(0.1, 2.0, n)         # Superficial Gas Velocity 
dp = np.random.uniform(50e-6, 500e-6, n)    # Particle Diameter
rho_p = np.random.uniform(1000, 3000, n)    # Partcile Density
H = np.random.uniform(0.2, 1.0, n)          # Bed Height
mu_g = 1.8e-5                               # Dynamic viscosity of gas
rho_g = 1.2                                 # Gas Density
g = 9.81                                    # Gravity
```

## Fluidized Bed Theory Used to Generate the Dataset

A. Minimum Fluidization Velocity ($U_{mf}$): The minimum fluidization velocity is the gas velocity at which the bed of particles starts to behave like a fluid. It is a critical point where the particles are no longer in a packed state but are suspended in the gas phase.

We use the Wen & Yu correlation (a simplified empirical model for gas-solid systems) to estimate Umf based on particle and gas properties:

Archimedes number: A dimensionless number that characterizes the buoyancy of particles in the gas. It's related to particle size, density, and gas properties.
    
$$Ar = \frac{d_p^3 \rho_g (\rho_p− \rho_g) \cdot g} {\mu_g^2}$$


Where:

$d_p​$ : Particle diameter ($m$)

$\rho_g$ ​: Gas density ($kg/m^3$)

$\rho_g$ : Particle density ($kg/m^3$)

$g$: Acceleration due to gravity ($m/s^2$)

$\mu_g$​: Gas viscosity ($Pa.s$)

The Reynolds number for the minimum fluidization is calculated from the Archimedes number. This gives an estimate for the minimum fluidization velocity $U_{mf}$​:

$${Re}_{mf} = \sqrt{33.72+0.0408⋅Ar} − 33.7$$

Then, $U_{mf}$ is calculated using the formula:

$$U_{mf} = \frac{{Re}_{mf} \mu_g}{\rho_g d_p}$$

Where:

$U_{mf}$​ is the minimum fluidization velocity ($m/s$).

B. Pressure Drop ($dP$):

The pressure drop across the bed is primarily driven by the density difference between the solid and gas phases, the bed height, and the superficial gas velocity. The bed undergoes two main fluidization regimes:

Dilute Fluidization ($Ug > Umf$): In this case, the pressure drop dPdP is calculated as a function of the superficial gas velocity:

$$dP=(\rho_p− \rho_g) g H (U_g > U_{mf})$$
 

Dense/Slugging Fluidization ($U_g < U_{mf}$): When the gas velocity is lower than the minimum fluidization velocity, particles form a packed bed and the pressure drop increases with the velocity in a more complex manner, which is approximated in the model by adjusting for gas velocity in the range below $U_{mf}$:

$$dP=(\rho_p− \rho_g) g H (U_g \leq U_{mf})⋅\left( \frac{U_g}{U_{mf}} \right)$$


C. Solid Holdup ($\varepsilon_s$):

The solid holdup ($\varepsilon_s$) is the fraction of the bed volume occupied by the solid particles. As the gas velocity increases, the bed becomes more fluidized, and the solid particles tend to occupy less of the bed volume. The relationship is approximated as:

$$\varepsilon_s =clip \left(0.6− 0.25 \frac{U_g} {U_{mf} + 1e−5},0.1,0.6 \right)$$

Where:

The function clip ensures that the solid holdup stays within a physically meaningful range, i.e., between $0.1$ and $0.6$.

Summary of Key Fluidized Bed Relationships

$U_{mf}$ (Minimum Fluidization Velocity) depends on particle properties, gas properties, and bed geometry.

Pressure Drop ($dP$) is calculated based on gas velocity relative to Umf.

Solid Holdup ($\varepsilon_s$) is a function of gas velocity and reflects the bed's fluidized state.

These simplified relationships help generate synthetic data for the model, which can then be used to train surrogate models that predict the pressure drop $dP$ given various operational conditions.


```python

# Fluidized bed theory
Ar = (dp**3 * rho_g * (rho_p - rho_g) * g) / (mu_g**2)                   # Archimedes number
Re_mf = (33.7**2 + 0.0408 * Ar)**0.5 - 33.7                              # Reynolds number 
Umf = Re_mf * mu_g / (rho_g * dp)                                        # Minimum Fluidisation Velocity                                              
dP = (rho_p - rho_g) * g * H * (Ug > Umf) + (rho_p - rho_g) * g * H * (Ug <= Umf) * (Ug / Umf) # Pressure Drop
eps_s = np.clip(0.6 - 0.25 * (Ug / (Umf + 1e-5)), 0.1, 0.6)              # Solid Holdup (clip ensures the range 0.1 to 0.6)   

```


```python
#%% DataFrame
data = pd.DataFrame({
    'Ug': Ug,
    'dp': dp * 1e6,  # microns
    'rho_p': rho_p,
    'H': H,
    'Umf': Umf,
    'dP': dP,
    'eps_s': eps_s
})

```


```python
#%% Preprocessing
features = ['Ug', 'dp', 'rho_p', 'H']
target = 'dP'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
#%% Train models
lr = LinearRegression().fit(X_train_scaled, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=20000, random_state=0).fit(X_train_scaled, y_train)
```


```python
#%% Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_mlp = mlp.predict(X_test_scaled)
```


```python
#%% Evaluation
def print_metrics(name, y_true, y_pred):
    print(f"{name} R2: {r2_score(y_true, y_pred):.3f}, MSE: {mean_squared_error(y_true, y_pred):.2e}")

print_metrics("Linear", y_test, y_pred_lr)
print_metrics("Random Forest", y_test, y_pred_rf)
print_metrics("MLP", y_test, y_pred_mlp)
```

    Linear R2: 0.949, MSE: 1.78e+06
    Random Forest R2: 0.990, MSE: 3.45e+05
    MLP R2: 1.000, MSE: 1.63e+04



```python
#%% Feature importances (Random Forest)
importances = rf.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Feature Importances - Random Forest")
plt.show()
```


    
![png](/assets/notebooks/PartiNet_v1_FluBedSurrogate_files/PartiNet_v1_FluBedSurrogate_12_0.png)
    



```python
#%% Prediction vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual dP")
plt.ylabel("Predicted dP")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_mlp, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual dP")
plt.ylabel("Predicted dP")
plt.title("MLP: Actual vs Predicted")
plt.grid(True)
plt.show()
```


    
![png](/assets/notebooks/PartiNet_v1_FluBedSurrogate_files/PartiNet_v1_FluBedSurrogate_13_0.png)
    



    
![png](/assets/notebooks/PartiNet_v1_FluBedSurrogate_files/PartiNet_v1_FluBedSurrogate_13_1.png)
    



```python

```
