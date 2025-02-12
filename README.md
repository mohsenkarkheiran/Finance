# European Options Pricing and Risk Management Using AI-Driven Models and Traditional Numerical Methods
## 1. Project Overview

The aim of this project is to develop a comprehensive solution for pricing European options and managing associated financial risks by integrating advanced AI models and traditional numerical methods. European options are financial derivatives with significant importance in modern financial markets, and accurately pricing these instruments is crucial for effective risk management.

This project will involve developing two types of models:

### Type A: AI-Driven Models <br />
  Physics-Informed Neural Networks (PINNs) <br />
  Recurrent neural nets (RNNs), eg. LSTM and GRU. <br />

### Type B: Traditional Numerical Methods <br />
  Monte Carlo Simulation<br />
  Time series analysis techniques for financial data<br />
  
These approaches will privde simple AI based methods for finding the option prices .

## 2. Objectives

Develop a Pricing Model for European Options: 

#### 1.Combine the traditional numerical methods and RNN for analyzing time serier with PINN and Q-learning for pricing European options. 
This includes implementing Black-Scholes as a benchmark model. <br />
##### 1.a. Finding trends, seasonality and forcasting using ARIMA models,
Download, preprocess the data, and try to forecast the future Stock prices. Asses the validity of ARIMA models. <br />
##### 1.b. Use Recurrent Neural Nets for stock price forecasting.
We will specifically use LSTM and GRU models.
##### 1.c. Check for volatility clustering.
Check whether the volatilities are propagating in time or they are just constants. This data used in geometrical Brownian motion to provide the third way for stock price forecasting. <br />
#### 2.Risk Management Metrics and Implied Volatility using PINN: 
Calculate Greeks (Delta, Gamma, Vega, Theta, and Rho) to quantify sensitivities and potential risk factors for options portfolios. <br />
#### 3.Implement the Fitted Q-Itteration for finding fair option price: 
The works of Igor Halberin (arXiv:1712.04609) is used to find the effect of variance backpropagation in Option pricing. <br/>
#### 4.GitHub Documentation and Visualization: <br />
Ensure all code, results, and documentation are well-structured for easy navigation on GitHub. Include visualizations to illustrate the performance and comparison of models.

## 3. Methodology

### .Forecasting:
After preprocessing the raw data, three methods are used to forecase the future stock prices up to maturity time (given by the user):
1. ARIMA with appropriate paramters, found by ACF/PACF graphs and ADF test.
2. RNN Networks (LSTM and GRU layers) used for time-series predictions of underlying asset prices.
3. The GARCH results are used to in GBM to forecast range of future prices.

### .PINNs: 
Implement a PINN to solve the Black-Scholes partial differential equation (PDE). PINNs will leverage the known physics underlying the option pricing formula to enhance learning accuracy and stability. In addition, one can use PINN's inverse parameter solving methods to find the implied volatility. On the other hand, PINN can be used to solve Black-Sholdes equation for non-constant volatility fed by GARCH. <br />

### .RL:
In arXiv:1712.04609, a refined version (relative to Black-Scholes model) for option price is suggested that includes the contribution of the future price uncertainty. Surprisingly, this "fair" option price satisfy the Bellman equations in reinforcement learning. The method suggested in this paper, uses Monte-Carlo methods to compute the Variance of portfolio values in future times, and then used Q-learning to find the best hedging strategy and fair option price at the same time. In this part the implied volatility if given by PINN. Moreover, we will extensively use the Halpern's course material/codes.

### .Comparison:
The option price provided by exact Black-Scholes formula, PINN's solution, and Q-learning method will be compared.

## 4. Project Structure

The project will be structured as follows:

1. Data Collection & Preprocessing: Acquire historical options data and preprocess for both training and evaluation. Use open datasets (such as Yahoo Finance).
2. Underlying asset analysis: ARIMA, RNN and GARCH models.
3. AI Models option pricing: PINN and Q-learning
4. Results Visualization: Create visualizations for price predictions.
5. Documentation and GitHub Portfolio: Ensure all code is modular, well-documented, and complemented with a project overview, model explanations, and instructions for reproducibility.



## 5. Deliverables

#### Codebase: 
Modular, well-documented code with separate folders for data processing, models, and visualization.
#### Documentation: 
Comprehensive README with explanations of model choice, usage instructions, and a summary of findings.
Reports: Detailed analysis and comparison report, highlighting which model performs best under specific conditions.

## 6. Conclusion

This project will contribute to the financial research community by exploring the AI-driven models for European options pricing and risk management. The GitHub repository will be a resource for practitioners and researchers looking to apply advanced AI techniques or traditional methods to option pricing problems, with reproducible, high-quality code and thorough documentation.
