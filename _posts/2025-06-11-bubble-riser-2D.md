---
layout: post
title: "Bubble rise through water in 2D using Basilisk"
math: true
date: 2025-06-11 08:00:00 +0200
categories: [projects]
tags: [cfd, vof, ml, basilisk, multiphase]
---

# Predicting Bubble Rise Velocity in Liquids using ML and Basilisk

The rise of a gas bubble in a liquid is a classic multiphase flow problem, rich in fluid mechanics and practical relevance — from boiling and cavitation to bioreactors and nuclear cooling.

In this project, we simulate 2D bubbles rising in a quiescent liquid using the **Volume of Fluid (VOF)** method in **Basilisk**, and then train a machine learning model to predict the terminal rise velocity from physical parameters.

---

## ⚙️ Governing Physics

The terminal rise velocity of a bubble in a liquid depends on several key parameters:

- **Bubble diameter** ($d_b$)
- **Liquid density** ($\rho_\ell$)
- **Gas density** ($\rho_g$)
- **Liquid viscosity** ($\mu$)
- **Surface tension** ($\sigma$)
- **Gravity** ($g$)

Dimensionless groups often used:
- **Eötvös number**:  
  $$
  Eo = \frac{(\rho_\ell - \rho_g) g d_b^2}{\sigma}
  $$
- **Morton number**:  
  $$
  Mo = \frac{g \mu^4 (\rho_\ell - \rho_g)}{\rho_\ell^2 \sigma^3}
  $$
- **Reynolds number** (based on terminal velocity):  
  $$
  Re = \frac{\rho_\ell v_t d_b}{\mu}
  $$

The relationship between these numbers is nonlinear and often regime-dependent (spherical, ellipsoidal, skirted, etc.). Analytical formulas exist in limiting cases, but not in general.

---

## 🎯 Goal

We will simulate multiple bubble rise scenarios by varying key parameters and extract the **terminal velocity**. These data points will be used to train an ML model to learn the mapping:

$$
(d_b, \rho_\ell, \rho_g, \mu, \sigma) \longrightarrow v_t
$$

Once trained, the model should be able to **predict the rise velocity** of a bubble given physical parameters — without needing to run new CFD simulations.


