# Regime-Switching Models for Corporate Refinancing

## Project Overview

This project investigates whether regime-switching models can improve
corporate debt refinancing decisions by modeling changing interest-rate,
credit-spread, and financial-market volatility conditions.

A Hidden Markov Model (HMM) will be used to estimate latent market regimes.
Principal Component Analysis (PCA) will be explored to construct a common
volatility factor from multiple market indicators. Regime-conditional
simulation and a refinancing decision model will then compare refinancing
now against waiting for potentially more favorable conditions.

## Research Question

Can incorporating latent financial-market regimes improve refinancing
timing and reduce expected financing costs relative to conventional
decision rules?

## Data Sources

- FRED: historical economic and financial market time series.
- SEC EDGAR: corporate financial statements and debt disclosures.
- Bloomberg: supplementary bond-market data, if available.

## Project Structure

- `data/`: downloaded datasets and data documentation.
- `dslc_documentation/`: numbered Jupyter notebooks documenting the
  analysis and modeling workflow.
- `dslc_documentation/functions/`: reusable Python functions.

## Analysis Workflow

1. Data acquisition and auditing.
2. Exploratory data analysis.
3. Feature engineering and PCA.
4. Hidden Markov Model estimation and regime interpretation.
5. Refinancing cost modeling and decision analysis.
6. Out-of-sample evaluation and backtesting.

## Reproducibility

The analysis will document data sources, transformations, modeling
assumptions, and evaluation procedures. Raw data and credentials should
not be committed to version control.
