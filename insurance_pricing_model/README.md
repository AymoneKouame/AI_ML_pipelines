DSP Insurance Pricing Model
Project Documentation

Overview
A machine learning model to optimize insurance pricing for Amazon's Delivery Service Partners (DSPs). The model analyzes 15+ risk factors to predict optimal premium pricing and identify business opportunities.
Goal: Build a data-driven pricing system that maximizes profitability while remaining competitive in the market.

What It Does
Core Functions

Predicts insurance premiums based on DSP characteristics and risk factors
Identifies high-risk vs. profitable customers for targeted pricing
Analyzes market competition and suggests pricing strategies
Provides business recommendations with projected financial impact

Key Features

5 different machine learning models (Random Forest performs best)
Risk segmentation and profitability analysis
Competitive pricing scenarios
Portfolio optimization recommendations


Data & Features
Input Data (15 variables)

Geographic: Region, urban density, weather risk
Economic: GDP per capita, unemployment rate
Operational: Fleet size, driver experience, daily packages
Performance: Delivery success rate, customer satisfaction, safety incidents
Market: Competitor density, market share
Claims: Historical claims ratio, average claim amount

Target Variable

Annual Insurance Premium: The price we charge each DSP


Model Results
Performance
ModelAccuracy (R²)Error (RMSE)StatusRandom Forest89.1%$2,724BestGradient Boosting89.6%$2,651WinnerRidge Regression84.7%$3,234GoodLinear Regression84.5%$3,247Baseline
Most Important Risk Factors

Historical Claims (25%) - Past claims predict future risk
Safety Incidents (16%) - More accidents = higher premiums
Fleet Size (13%) - Larger fleets = more exposure
Urban Density (10%) - City driving = higher risk
Weather Risk (9%) - Bad weather = more claims