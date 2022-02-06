module ForwardCurveSmoother

using DataFrames
using CSV
using Dates
using JuMP
using OSQP
using GLM
using GLPK
using TimeSeriesInterface
using Lasso

include("structs.jl")
include("optimization.jl")
include("utils.jl")

function fit_structural_model(contracts::DataFrame, spot::DataFrame, structural_parameters::StructuralParameters)

    # Creating input and running the smoothing algorithm
    input_structural, coeficients = create_structural_model_input(contracts, spot, 
                                                                  structural_parameters)
    output_structural = forward_structural_model(input_structural)

    # Handle results
    reconstructed_prices = handle_data_frame_of_reconstructed_prices(contracts, output_structural.reconstructed_forward_prices)
    elementary_prices    = output_structural.structural_dinamic .+ output_structural.structural_error
    elementary_errors    = output_structural.structural_error

    number_of_maturities = size(elementary_prices, 2)
    timestamps           = contracts[:, 1]
    names_columns        = [:timestamps; [Symbol("maturity_$i") for i in 1:number_of_maturities]]

    df_elementary_prices = DataFrame(hcat(timestamps, elementary_prices), names_columns)
    df_elementary_errors = DataFrame(hcat(timestamps, elementary_errors), names_columns)

    # Saving CSVs on the current path
    @info("Saving reconstructed_prices.csv...")
    CSV.write("Results/reconstructed_prices.csv", reconstructed_prices; delim = ';')
    @info("Saving elementary_prices.csv...")
    CSV.write("Results/elementary_prices.csv", df_elementary_prices; delim = ';')
    @info("Saving elementary_errors.csv...")
    CSV.write("Results/elementary_errors.csv", df_elementary_errors; delim = ';')

    return input_structural, output_structural
end

end
