function create_structural_model_input(contracts::DataFrame, spot::DataFrame, 
                                            structural_parameters::StructuralParameters)

    exogenous_variables       = handle_f0(spot)
    parameters                = handle_parameters(structural_parameters, 
                                    exogenous_variables["f0_spot"])
    vec_contracts_by_day      = read_ContractsByDay(contracts)
    dict_parametric_functions = create_dict_parametric_functions()
    maximum_maturity          = get_maximum_maturity(parameters)

    return create_structural_model_input_struct(parameters,
                vec_contracts_by_day,
                exogenous_variables,
                dict_parametric_functions,
                maximum_maturity)
    end

function create_structural_model_input_struct(parameters::Dict,
          vec_contracts_by_day::Vector,
          exogenous_variables::Dict,
          dict_parametric_functions::Dict{String, Vector{Function}},
          maximum_maturity::Int64)

    timestamps          = date_of_trades(vec_contracts_by_day)
    exogenous_variables = handle_structural_model_structure(exogenous_variables, parameters, timestamps)
    X, X_var_names      = create_structural_model_X(exogenous_variables,
                    maximum_maturity,
                    dict_parametric_functions,
                    has_f0_spot = true)

    return StructuralModelInputs(parameters,
                        vec_contracts_by_day,
                        X), X_var_names

end

function handle_f0(spot::DataFrame)
    name       = "f0_spot"
    timestamps = convert(Vector{DateTime}, spot.timestamps)
    vals       = convert(Vector{Float64}, spot.vals)

    time_series = TimeSeriesInterface.TimeSeries(name,
                timestamps,
                vals)

    return Dict{String, TimeSeries{Float64}}("f0_spot" => time_series)
end

function handle_parameters(structural_parameters::StructuralParameters,
                                f0_spot::TimeSeries{Float64})

    parameters = Dict{String, Any}()
    parameters["interest_yield"]             = structural_parameters.interest_yield
    parameters["intercept"]                  = structural_parameters.intercept
    parameters["intercept_varying_maturity"] = structural_parameters.intercept_varying_maturity
    parameters["seasonality_trading_date"]   = structural_parameters.seasonality_trading_date
    parameters["seasonality_delivery_date"]  = structural_parameters.seasonality_delivery_date
    parameters["run_lasso"]                  = structural_parameters.run_lasso
    parameters["minimum_maturity"]           = structural_parameters.minimum_maturity
    parameters["λ"]                          = structural_parameters.λ
    parameters["f0_spot"]                    = f0_spot

    return parameters
end

function create_dict_parametric_functions()
    dict_parametric_functions = Dict{String,Vector{Function}}()
    dict_parametric_functions["seasonality_delivery_date"] = Function[
                                                                f_sin_t_j_1_harm,
                                                                f_cos_t_j_1_harm,
                                                                #f_sin_t_j_2_harm,
                                                                #f_cos_t_j_2_harm
                                                                ]
    dict_parametric_functions["seasonality_trading_date"] = Function[
                                                                f_sin_t_1_harm,
                                                                f_cos_t_1_harm,
                                                                #f_sin_t_2_harm,
                                                                #f_cos_t_2_harm
                                                                ]
    dict_parametric_functions["intercept_varying_maturity"] = Function[
                                                                f_sqrt_j_1,f_sqrt_j_2,f_sqrt_j_3
                                                                ]
    dict_parametric_functions["spot"] = Function[
                                            f_sqrt_j_0, f_sqrt_j_1,f_sqrt_j_2
                                            ]
    return dict_parametric_functions
end

function get_maximum_maturity(parameters::Dict)
    return parameters["minimum_maturity"]
end

function handle_structural_model_structure(exogenous_variables::Dict, 
       parameters::Dict, 
       timestamps::Vector)
    ones_aux_ts = TimeSeries(
                        "ones aux ts",
                        timestamps,
                        ones(length(timestamps))
                        )
    if parameters["seasonality_delivery_date"]
        exogenous_variables["seasonality_delivery_date"] = ones_aux_ts
    end
    if parameters["seasonality_trading_date"]
        exogenous_variables["seasonality_trading_date"] = ones_aux_ts
    end
    if parameters["intercept_varying_maturity"]
        exogenous_variables["intercept_varying_maturity"] = ones_aux_ts
    end
    if parameters["intercept"]
        exogenous_variables["intercept"] = ones_aux_ts
    end
    return exogenous_variables
end

function create_structural_model_X(structural_exogenous::Dict,
                                        maximum_maturity::Int,
                                        dict_parametric_functions::Dict;
                                        has_f0_spot = true)
    names_exogenous = get_alphabetically_ordered_keys(structural_exogenous)
    names_exogenous = handle_exogenous_names(names_exogenous)
    X = nothing
    X_columns_names = String[]
    for (n, exogenous) in enumerate(names_exogenous)
        series = structural_exogenous[exogenous]
        parametric_functions = handle_exogenous_parametric_functions(
                                        exogenous,
                                        dict_parametric_functions
                                        )
        x = create_exogenous(
                        series, maximum_maturity, 
                        parametric_functions, 
                        has_f0_spot = has_f0_spot
                        )
        x_columns_names = exogenous_name(
                                exogenous, parametric_functions
                                )
        X = append_X(X, x)
        X_columns_names = vcat(X_columns_names, x_columns_names)
    end
    return X, X_columns_names
end

function exogenous_name(exogenous::String, parametric_function::Function)
    if haskey(DICT_FUNCTIONS_NAMES, parametric_function)
        return exogenous * DICT_FUNCTIONS_NAMES[parametric_function]
    end
    return exogenous * "_customized_function"
end

function exogenous_name(exogenous::String, parametric_functions::Vector)
    names = String[]
    for (f, func) in enumerate(parametric_functions)
        push!(names, exogenous_name(exogenous, func))
    end
    return names
end

function get_alphabetically_ordered_keys(dict::Dict)
    ks = String[]
    for k in keys(dict)
        push!(ks, k)
    end
    return sort(ks)
end

function handle_exogenous_names(names_exogenous)
    names_exogenous = remove(names_exogenous, "f0_spot")
    return names_exogenous
end

function handle_exogenous_parametric_functions(exogenous::String,
           dict_parametric_functions::Dict)
    parametric_functions = Function[f_sqrt_j_0]
    if exogenous == "spot"
        parametric_functions = dict_parametric_functions["spot"]
    end
    if exogenous == "seasonality_delivery_date"
        parametric_functions = dict_parametric_functions["seasonality_delivery_date"]
    end
    if exogenous == "seasonality_trading_date"
        parametric_functions = dict_parametric_functions["seasonality_trading_date"]
    end
    if exogenous == "intercept_varying_maturity"
        parametric_functions = dict_parametric_functions["intercept_varying_maturity"]
    end
    return parametric_functions
end

function create_exogenous(series::TimeSeries, 
                            maximum_maturity::Int, 
                            parametric_function::Function;
                            has_f0_spot = true)

    T = length(series.timestamps)
    J = maximum_maturity
    x = Matrix{Float64}(undef, T, J + has_f0_spot)
    for t in 1:T
        for j in 1:J
            x[t,j] = series.vals[t] * parametric_function(series.timestamps[t], j)
        end
        if has_f0_spot
            x[t,J + 1] = series.vals[t] * parametric_function(series.timestamps[t], 0)
        end
    end

    return x
end

function create_exogenous(series::TimeSeries, 
                            maximum_maturity::Int, 
                            parametric_functions::Vector{Function};
                            has_f0_spot=false)
    T = length(series.timestamps)
    J = maximum_maturity
    num_coefs = length(parametric_functions)
    x = Array{Float64}(undef, T, J + has_f0_spot, num_coefs)

    for (f, func) in enumerate(parametric_functions)
        x[:,:, f] = create_exogenous(
                            series,
                            maximum_maturity,
                            func,
                            has_f0_spot=has_f0_spot
                            )
    end
    return x
end

function handle_exogenous_names(names_exogenous)
    names_exogenous = remove(names_exogenous, "f0_spot")
    return names_exogenous
end

function remove(vec_string::Vector{String}, string)
    if string in vec_string
        vec_string = setdiff(
        vec_string, vec_string[findall(x -> x == "f0_spot", vec_string)]
        )
    end
    return vec_string
end

function append_X(X, 
                x::Array)
    if X == nothing
        return x
    end
    return cat(X, x, dims=3)
end

# Parametric Functions
f_sqrt_j_0(t::DateTime, j::Int) = sqrt(j)^0
f_sqrt_j_1(t::DateTime, j::Int) = sqrt(j)^1
f_sqrt_j_2(t::DateTime, j::Int) = sqrt(j)^2
f_sqrt_j_3(t::DateTime, j::Int) = sqrt(j)^3

f_sin_t_1_harm(t::DateTime, j::Int) = sin(2 * π * 1 * dayofyear(t) / 365)
f_cos_t_1_harm(t::DateTime, j::Int) = cos(2 * π * 1 * dayofyear(t) / 365)
#f_sin_t_2_harm(t::DateTime, j::Int) = sin(2 * π * 2 * dayofyear(t) / 365)
#f_cos_t_2_harm(t::DateTime, j::Int) = cos(2 * π * 2 * dayofyear(t) / 365)

f_sin_t_j_1_harm(t::DateTime, j::Int) = sin(2 * π * 1 * dayofyear(t + Day(j)) / 365)
f_cos_t_j_1_harm(t::DateTime, j::Int) = cos(2 * π * 1 * dayofyear(t + Day(j)) / 365)
#f_sin_t_j_2_harm(t::DateTime, j::Int) = sin(2 * π * 2 * dayofyear(t + Day(j)) / 365)
#f_cos_t_j_2_harm(t::DateTime, j::Int) = cos(2 * π * 2 * dayofyear(t + Day(j)) / 365)

DICT_FUNCTIONS_NAMES = Dict(
f_sqrt_j_0 => "_√j⁰",
f_sqrt_j_1 => "_√j¹",
f_sqrt_j_2 => "_√j²",
f_sqrt_j_3 => "_√j³",
f_sin_t_j_1_harm => "_sin_1_harmonic",
#f_sin_t_j_2_harm => "_sin_2_harmonic",
f_cos_t_j_1_harm => "_cos_1_harmonic",
#f_cos_t_j_2_harm => "_cos_2_harmonic",
f_sin_t_1_harm => "_sin_1_harmonic",
#f_sin_t_2_harm => "_sin_2_harmonic",
f_cos_t_1_harm => "_cos_1_harmonic",
#f_cos_t_2_harm => "_cos_2_harmonic"
)

function handle_data_frame_of_reconstructed_prices(contracts::DataFrame, reconstructed_forward_prices::Vector{ContractsByDay})

    reconstructed_contracts = deepcopy(Matrix(contracts))
    number_maturities       = length(reconstructed_forward_prices[1].contracts)
    number_observations     = length(reconstructed_forward_prices)
    number_cols_contracts   = size(contracts, 2)

    @assert number_observations == size(contracts, 1)

    for col in 2:3:number_cols_contracts - 2
        reconstructed_contracts[:, col] .= missing
    end

    for (t, contract_info) in enumerate(reconstructed_forward_prices)
        cont_maturity = 1 
        for col in 2:3:number_cols_contracts - 2
            reconstructed_contracts[t, col] = contract_info.contracts[cont_maturity].price
            cont_maturity += 1
        end
    end

    return DataFrame(reconstructed_contracts, names(contracts))
end