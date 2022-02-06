import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/ForwardCurveSmoother.jl")

# If the packages below are not yet installed, you must install it once using the following command:
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# After the installation, run the commands below to load the package.
using DataFrames
using CSV

spot     = CSV.read("Data/spot.csv", DataFrame)
contracts = CSV.read("Data/contracts.csv", DataFrame)

# Parameters used in the semi-parametric structural model
discount_factor            = 0.0 # discount factor used in the net present value
intercept                  = true # true or false, defining the existence of intercept in the forward curve equation 
intercept_varying_maturity = false # true or false for the existence of a time-varying intercept according to the following equation: sqrt(j) + sqrt(j)^2 + sqrt(j)^3
seasonality_trading_date   = true # true or false for seasonality in the trading dates (one harmonic trigonometric function) 
seasonality_delivery_date  = true # true or false for seasonality in the delivery dates (maturity) (one harmonic trigonometric function)
run_lasso                  = false # true or false to run AdaLasso to automatically select between the previous structures
minimum_maturity           = 2000 # maximum maturity to be available in the estimated forward curves
λ                          = 0.5 # objective function weight in the smoothing in the trading dates dimension

# Build a structure with the previous defined parameters
structural_parameters = ForwardCurveSmoother.StructuralParameters(discount_factor, intercept, intercept_varying_maturity,
                                                                            seasonality_trading_date, seasonality_delivery_date,
                                                                            run_lasso, minimum_maturity, λ)

# Run the semi-parametric structural model to estimate the forward curves.
# Additionally to the 'structural_parameters' built previously, two run the algorithm are necessary two CSV files, contracts.csv and spot.csv (template in folder 'Data').
# The first one contains the forward prices of contracts of different maturities and their correspondent delivery period, for different trading dates.
# The second, contains the spot prices for the same trading dates.

# The function output is two structs, one with the input and the other with the complete output of the model.
# Three CSV files are automatically saved in the folder 'Results': 
#   recontructed_prices.csv - CSV file with the prices estimated for each one of the contracts defined in contracts.csv;
#   elementary_prices.csv - CSV file with the estimated forward curve for each trading date. This curve is composed by the elementary prices, which are contracts
#   with the delivery period of a day. 
#   elementary_errors.csv - CSV file with the estimated smoothed errors.                                                                  
input_structural, output_structural = ForwardCurveSmoother.fit_structural_model(contracts, spot, structural_parameters);    
                                                                                                    

