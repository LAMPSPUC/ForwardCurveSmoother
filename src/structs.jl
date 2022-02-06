"""
    Defines the parameters used in the smoothing process.
"""
mutable struct StructuralParameters
    interest_yield::Float64
    intercept::Bool
    intercept_varying_maturity::Bool
    seasonality_trading_date::Bool
    seasonality_delivery_date::Bool
    run_lasso::Bool
    minimum_maturity::Int64
    λ::Float64
end

"""
    ContractInfo

Defines a negotiated forward contract.
"""
mutable struct ContractInfo
    trading_day::DateTime
    time_to_maturity::Int64 # H
    duration::Int64 # ΔT
    price::Float64 # F
end

"""
    ContractsByDay

Defines a vector of contracts traded in a trading day.
"""
mutable struct ContractsByDay
    trading_day::DateTime
    contracts::Vector{ContractInfo}
end

function number_of_contracts(vec_contracts_by_day::Vector{ContractsByDay})
    return number_of_contracts(vec_contracts_by_day[1])
end

function number_of_contracts(contracts_by_day::ContractsByDay)
    return length(contracts_by_day.contracts)
end

function date_of_trades(vec_contracts_by_day::Vector{ContractsByDay})
    dates = Vector{DateTime}(undef, length(vec_contracts_by_day))

    for (t, contracts_by_day) in enumerate(vec_contracts_by_day)
        dates[t] = contracts_by_day.trading_day
    end

    return dates
end


"""
    OptimizationInputs

parameters = Dict(
    "interest_yield" = 0.0
)
"""
mutable struct StructuralModelInputs
    parameters::Dict{String,Any}
    vec_contracts_by_day::Vector{ContractsByDay}
    X::Array{Float64,3}

    function StructuralModelInputs(parameters::Dict{String,Any},
                                   vec_contracts_by_day::Vector{ContractsByDay},
                                   X::Array{Float64,3})

        dates = date_of_trades(vec_contracts_by_day)

        T, J, L = size(X)
        if length(vec_contracts_by_day) != T
            error("X and vec_contracts_by_day must have same days")
        end
        
        return new(parameters, vec_contracts_by_day, X)
    end
end

function StructuralModelInputs(parameters::Dict{String,Any},
                                vec_contracts_by_day::Vector{ContractsByDay},
                                X::Nothing)

    max_maturity = maximum_maturity_from_vec_contracts_by_day(vec_contracts_by_day)
    max_maturity = max(max_maturity, parameters["minimum_maturity"])
    T = length(vec_contracts_by_day) 
    
    return return StructuralModelInputs(parameters, vec_contracts_by_day, zeros(T, max_maturity, 0))
end

mutable struct ThirdLevelSolveTime
    maturity_smooth::Float64
    time_smooth::Float64
    estimate_structural_error::Float64
end

mutable struct StructuralSolveTime
    ground_level::Float64
    first_level::Float64
    second_level::Float64
    third_level::ThirdLevelSolveTime
end

mutable struct StructuralModelOutputs
    glm_output::Any
    args_structural_expression::Dict
    daily_minimum_arbitrage::Vector{Float64}
    maturity_weigth::Float64
    time_weigth::Float64
    structural_error_obj_function::Float64
    structural_error::Matrix{Float64}
    structural_dinamic::Matrix{Float64}
    reconstructed_forward_prices::Vector{ContractsByDay}
    solve_time::StructuralSolveTime
    adalasso_model
    lasso_selected_variables_idx::Vector{Int64}
    minimum_arbitrage_by_contract::Vector
end


"""
    read_ContractsByDay(df::DataFrame)

Converts the input data frame to the adequate format used in the input struct.
"""
function read_ContractsByDay(df::DataFrame)
    vec_contracts_by_day = ContractsByDay[]
    for row in eachrow(df)
        contracs_by_day = get_contracts_by_day(row)
        push!(vec_contracts_by_day, contracs_by_day)
    end
    return vec_contracts_by_day
end

function get_contracts_by_day(row::DataFrameRow)
    trading_day = row[1]
    num_contracts = Int(length(row[2:end]) / 3)
    vec_contract_info = ContractInfo[]
    
    for contract = 1:num_contracts
        price                   = row[2 + 3 * (contract - 1)]
        start_delivering_period = row[3 + 3 * (contract - 1)]
        end_delivering_period   = row[4 + 3 * (contract - 1)]
        contract_info = get_contract_info(trading_day, start_delivering_period, end_delivering_period, price)
        push!(vec_contract_info, contract_info)
    end
    
    return ContractsByDay(trading_day, vec_contract_info)
end

contract_price(price) = ismissing(price) ? NaN : price
duration_between_dates_by_day(dt1::DateTime, dt2::DateTime) = Dates.days(convert(Dates.Day, (dt2 - dt1)))

function get_contract_info(trading_day::DateTime, 
                        start_delivering_period::DateTime, 
                        end_delivering_period::DateTime, 
                        price)
    time_to_maturity = duration_between_dates_by_day(trading_day, start_delivering_period)
    duration         = duration_between_dates_by_day(start_delivering_period, end_delivering_period)
    price            = contract_price(price)
    
    return ContractInfo(trading_day,
                    time_to_maturity, 
                    duration,
                    price)
end

