const QUAD_SOLVER = optimizer_with_attributes(OSQP.Optimizer, "verbose" => false, "max_iter" => 15000)
const LIN_SOLVER = optimizer_with_attributes(GLPK.Optimizer)
const TWO_PI_OVER_365 = 2 * π / 365

function calc_J(interest_yield::Number, maturity_interval::UnitRange{Int64})
    return sum((1 + interest_yield)^(-j) for j in maturity_interval)
end

function get_index_and_length_of_traded_contracts(contracts_by_day)
    pos_of_traded_contracts = Int[]
    for i = 1:length(contracts_by_day.contracts)
        if !isnan(contracts_by_day.contracts[i].price)
            push!(pos_of_traded_contracts, i)
        end
    end
    return pos_of_traded_contracts, length(pos_of_traded_contracts)
end

function construct_objective_function_from_adjustment(ω)
    objective = 0.0
    for t in 1:length(ω), adjust_var in ω[t]
        objective += adjust_var^2
    end
    return objective
end

function structural_expression(args_structural_expression, X)
    return sum(args_structural_expression[:beta] .* X)
end

function save_estimated_parameters!(args_structural_expression::Dict, 
                                    coeficients::Vector{Float64})
    args_structural_expression[:beta] = coeficients
    return
end

function create_F_stack(input::StructuralModelInputs)
    vec_contracts_by_day = input.vec_contracts_by_day
    T                    = length(date_of_trades(vec_contracts_by_day))
    I                    = number_of_contracts(vec_contracts_by_day)
    F_matrix             = Matrix(undef, T, I)
    contracts_idx_matrix = Int.(hcat([ones(T) * i for i = 1:I]...))

    for i in 1:I, t in 1:T
        F_matrix[t, i] = vec_contracts_by_day[t].contracts[i].price
    end

    F_stack             = vcat(F_matrix...)
    contracts_idx_stack = vcat(contracts_idx_matrix...)
    remove_missing_idx  = isnan.(F_stack) .== false
    F_stack             = F_stack[remove_missing_idx]
    contracts_idx_stack = contracts_idx_stack[remove_missing_idx]

    return F_stack, remove_missing_idx, contracts_idx_stack
end

function from_X_t_j_to_X_t_i(input::StructuralModelInputs)
    interest_yield       = input.parameters["interest_yield"]
    vec_contracts_by_day = input.vec_contracts_by_day
    X                    = input.X
    I                    = number_of_contracts(vec_contracts_by_day)
    (T, J, L)            = size(X)
    X_t_i_matrix         = Array{Any,3}(undef, T, I, L)
    for (t, contract_by_day) in enumerate(vec_contracts_by_day)
        for (i, contract_info) in enumerate(contract_by_day.contracts)
            duration         = contract_info.duration
            time_to_maturity = contract_info.time_to_maturity
            maturity_range   = time_to_maturity:time_to_maturity + duration - 1 
            for l = 1:L
                X_t_i_matrix[t, i, l] = sum(
                    X[t, j, l] * (1 + interest_yield)^(-j) for j in maturity_range
                ) / calc_J(interest_yield, maturity_range)
            end
        end
    end
    return X_t_i_matrix
end

function stack_X_matrix(X_t_i_matrix::Array)
    (T, I, L) = size(X_t_i_matrix)
    X_t_i_stack = Matrix{Float64}(undef, T * I, L)
    for i in 1:I, t in 1:T, l in 1:L
        X_t_i_stack[T * (i - 1) + t, l] = X_t_i_matrix[t, i, l]
    end
    return X_t_i_stack
end

function create_X_stack(input::StructuralModelInputs, remove_missing_idx::BitArray)
    X_t_i_matrix = from_X_t_j_to_X_t_i(input)
    X_t_i_stack  = stack_X_matrix(X_t_i_matrix)
    
    return  X_t_i_stack[remove_missing_idx, :]
end

function estimate_coeficients(F_stack::Vector, X_stack::Matrix)
    glm_model             = lm(X_stack, F_stack)
    estimated_coeficients = coef(glm_model)
    return estimated_coeficients, glm_model
end

"""
    Verify if X matrix has a column with all entries 'ones'.
        If true: return the matrix without this column and the idx
        If false: return the X matrix and a black vector
"""
function check_for_intercept(X::Matrix)
    has_intercept = false
    num_intercepts = 0
    for i = 1:size(X, 2)
        if all(X[:, i] .== 1)
            num_intercepts += 1
            has_intercept = true
        end
    end
    if num_intercepts > 1
        @error("Number of intercepts in structural model is greater than one. Revise the structural variables")
    end
    if has_intercept
        lin, col = size(X)
        X_wout_intercept = Matrix{Float64}(undef, lin, col - 1)
        (j, removed_idx) = (1, 0)
        for i = 1:size(X, 2)
            if !all(X[:, i] .== 1)
                X_wout_intercept[:,j] = X[:, i]
                j += 1
            else
                removed_idx = i
            end
        end
        return X_wout_intercept, removed_idx
    end
    return X, []
end

function structural_lasso_selection(input::StructuralModelInputs,
                                    X_stack::Matrix{Float64}, 
                                    F_stack::Vector{Float64})
    if input.parameters["run_lasso"]
        println("Selecting variables through AdaLasso.")
        return structural_lasso_selection(X_stack, F_stack)
    else
        println("No variable selection.")
        all_idx = collect(1:size(X_stack, 2))
        return "", all_idx
    end
end

function structural_lasso_selection(X_stack::Matrix{Float64}, y::Vector{Float64})
    X, rmv_idx = check_for_intercept(X_stack)
    T, L = size(X)
    cp_idx = collect(1:L)
    LASSO = try
        Lasso.fit(LassoModel, X, y; select=MinBIC())    
    catch
        "Failed to converge Lasso algorithm in structural model. Running structural model only with intercept"
    end
    if typeof(LASSO) == String
        @warn(LASSO)
        return "", setdiff(cp_idx, rmv_idx)
    end
    LASSO_coef_without_intercept = coef(LASSO)[2:end]
    cp_idx_nonzeros = cp_idx[abs.(LASSO_coef_without_intercept) .> 0]
    pen = abs.(LASSO_coef_without_intercept[cp_idx_nonzeros]).^(-1)
    X_ada = X[:, cp_idx_nonzeros]

    # AdaLasso
    AdaLASSO = try
        Lasso.fit(LassoModel, X_ada, y, penalty_factor=pen, select=MinBIC())
    catch
        "Failed to converge AdaLasso algorithm in structural model. Running structural model only with intercept"
    end
    if typeof(AdaLASSO) == String
        @warn(AdaLASSO)
        return "", setdiff(cp_idx, rmv_idx)
    end
    AdaLASSO_coef_without_intercept = coef(AdaLASSO)[2:end]
    ada_idx_nonzeros = AdaLASSO_coef_without_intercept .!= 0
    AdaLASSO_coef_without_intercept = AdaLASSO_coef_without_intercept[ada_idx_nonzeros]
    variables_idx = cp_idx_nonzeros[ada_idx_nonzeros]
    variables_idx_final = union(setdiff(collect(1:size(X_stack, 2)), rmv_idx)[variables_idx], rmv_idx)
    return AdaLASSO, sort(variables_idx_final)
end
    
"""
"""
function first_level_estimate_parameters(input::StructuralModelInputs)
    println(stdout, "Estimating seasonality and explanatory coefficients.")
    args_structural_expression = Dict{Symbol,Any}()
    # Create F_stack
    F_stack, remove_missing_idx, contracts_idx = create_F_stack(input)
    # Create X_stack
    X_stack = create_X_stack(input, remove_missing_idx)
    struct_AdaLASSO, select_variables_idx = structural_lasso_selection(input, X_stack, F_stack)
    X_stack = X_stack[:, select_variables_idx]
    estimated_coeficients, glm_model = estimate_coeficients(F_stack, X_stack)

    save_estimated_parameters!(args_structural_expression, estimated_coeficients)
    return (struct_AdaLASSO, select_variables_idx, args_structural_expression, glm_model)
end

function maximum_maturity_from_vec_contracts_by_day(vec_contracts_by_day::Vector{ContractsByDay})
    maximum_maturity = 1
    for contracts_by_day in vec_contracts_by_day, contract_info in contracts_by_day.contracts
        if contract_info.time_to_maturity + contract_info.duration - 1 > maximum_maturity
            maximum_maturity = contract_info.time_to_maturity + contract_info.duration - 1
        end
    end
    return maximum_maturity
end

function maximum_maturity_from_contracts_by_day(contracts_by_day::ContractsByDay)
    maximum_maturity = 1
    for contract_info in contracts_by_day.contracts
        if contract_info.time_to_maturity + contract_info.duration - 1 > maximum_maturity
            maximum_maturity = contract_info.time_to_maturity + contract_info.duration - 1
        end
    end
    maximum_maturity
end

function create_second_level_sub_problem_variables!(model::JuMP.Model, 
                                                    maximum_maturity::Int,
                                                    N::Int)
    @variable(model, ε[1:maximum_maturity])
    @variable(model, ζ[1:N])
    @variable(model, ζ_abs[1:N])
    return
end

function create_second_level_sub_problem_constraints!(model::JuMP.Model, 
                                                      args_structural_expression::Dict{Symbol,Any},
                                                      X_t::Matrix,
                                                      contracts_by_day::ContractsByDay, 
                                                      N::Int, 
                                                      interest_yield::Number,
                                                      index_of_traded_contracts::Vector{Int})
    for i = 1:N
        contract_info              = contracts_by_day.contracts[index_of_traded_contracts[i]]
        price                      = contract_info.price
        contract_maturity_interval = contract_info.time_to_maturity:contract_info.time_to_maturity + contract_info.duration - 1
        
        @constraint(model, contract_info.price == (
            sum(
                (structural_expression(args_structural_expression, X_t[j,:]) + model[:ε][j]) * 
                (1 + interest_yield)^(-j) for j in contract_maturity_interval
            ) / 
            calc_J(interest_yield, contract_maturity_interval)
            ) + 
                model[:ζ][i]
            )
        @constraint(model, model[:ζ_abs][i] >=  model[:ζ][i])
        @constraint(model, model[:ζ_abs][i] >= -model[:ζ][i])
    end
    return
end

function create_second_level_sub_problem_objective!(model)        
    return @objective(model, Min, sum(model[:ζ_abs]))
end

function sub_problem_daily_minimum_arbitrage(args_structural_expression::Dict{Symbol,Any}, 
                                             X_t::Matrix,
                                             contracts_by_day::ContractsByDay,
                                             interest_yield::Number)

    maximum_maturity = maximum_maturity_from_contracts_by_day(contracts_by_day)
    index_of_traded_contracts, N = get_index_and_length_of_traded_contracts(contracts_by_day)

    if N != 0
        model = Model(LIN_SOLVER)
        create_second_level_sub_problem_variables!(model, maximum_maturity, N)
        create_second_level_sub_problem_constraints!(model, args_structural_expression, 
                                                     X_t, contracts_by_day, N, 
                                                     interest_yield,
                                                     index_of_traded_contracts)
        create_second_level_sub_problem_objective!(model)
        
        optimize!(model)

        if (termination_status(model) != MOI.OPTIMAL)
            error("O problema não foi resolvido até a otimalidade. $(termination_status(model))")
        end

        return objective_value(model)
    else
        return 0.0
    end
end

function second_level_daily_minimum_arbitrage(args_structural_expression::Dict{Symbol,Any}, 
                                              input::StructuralModelInputs)
    println(stdout, "Minimum arbitrage calculation.")
    vector_of_daily_minimum_arbitrage = Vector{Float64}(undef, length(input.vec_contracts_by_day))
    X = input.X
    for (t, contracts_by_day) in enumerate(input.vec_contracts_by_day)
        daily_minimum_arbitrage = sub_problem_daily_minimum_arbitrage(
                                                args_structural_expression,
                                                X[t,:,:],
                                                contracts_by_day, 
                                                input.parameters["interest_yield"]
                                            )
        vector_of_daily_minimum_arbitrage[t] = daily_minimum_arbitrage
    end

    return vector_of_daily_minimum_arbitrage
end

function handle_f0_constraint!(model::JuMP.Model,
                               input::StructuralModelInputs, 
                               args_structural_expression::Dict{Symbol,Any},
                               X_t_j::Vector,
                               t::Int)
    if haskey(input.parameters, "f0_spot")
        @constraint(
                model, 
                structural_expression(args_structural_expression, X_t_j)
                + 
                model[:ε][t,0]
                == input.parameters["f0_spot"].vals[t]
                )
    end
    return 
end

maturity_0(X_t::Matrix) = X_t[end, :]

function third_level_common_constraints!(model::JuMP.Model,
                                         input::StructuralModelInputs, 
                                         args_structural_expression::Dict{Symbol,Any})
    T              = length(input.vec_contracts_by_day)
    interest_yield = input.parameters["interest_yield"]
    ζ              = Vector{Vector{VariableRef}}(undef, T)
    ζ_abs          = Vector{Vector{VariableRef}}(undef, T)
    X              = input.X
    arbitrage_constraint = 0.0
    for (t, contracts_by_day) in enumerate(input.vec_contracts_by_day)
        index_of_traded_contracts, N = get_index_and_length_of_traded_contracts(contracts_by_day)
        ζ[t]     = @variable(model, [1:N])
        ζ_abs[t] = @variable(model, [1:N])
        handle_f0_constraint!(
            model,
            input,
            args_structural_expression,
            maturity_0(X[t,:,:]),
            t)

        for i = 1:N
            contract_info              = contracts_by_day.contracts[index_of_traded_contracts[i]]
            contract_maturity_interval = contract_info.time_to_maturity:contract_info.time_to_maturity + 
                                         contract_info.duration - 1

            @constraint(model, contract_info.price == (
                sum(
                    (structural_expression(args_structural_expression, X[t, j, :]) + model[:ε][t,j]) * 
                    (1 + interest_yield)^(-j) for j in contract_maturity_interval
                ) / 
                calc_J(interest_yield, contract_maturity_interval)
                ) + 
                    ζ[t][i]
                )
            @constraint(model, ζ_abs[t][i] >=  ζ[t][i])
            @constraint(model, ζ_abs[t][i] >= -ζ[t][i])
            arbitrage_constraint += ζ_abs[t][i]
        end
    end
    return arbitrage_constraint
end

function sub_problem_smooth_through_maturiy(args_structural_expression::Dict{Symbol,Any}, 
                                            input::StructuralModelInputs, 
                                            vector_of_daily_minimum_arbitrage::Vector{Float64})
    println(stdout, "Getting the smoothing in maturities weight.")

    T = length(input.vec_contracts_by_day)
    maximum_maturity_contracts = maximum_maturity_from_vec_contracts_by_day(input.vec_contracts_by_day)
    maximum_maturity = max(maximum_maturity_contracts, input.parameters["minimum_maturity"])

    model = Model(QUAD_SOLVER)
    # structural error
    @variable(model, ε[0:T, 0:maximum_maturity])
    # smooth through maturity
    @variable(model, Γ[1:T, 1:maximum_maturity])
    arbitrage_constraint = third_level_common_constraints!(model, input, args_structural_expression)
    @constraint(model, [t = 1:T, j = 1:maximum_maturity - 1], Γ[t, j] == ε[t, j + 1] - 2 * ε[t, j] + ε[t, j - 1])
    @constraint(model, arbitrage_constraint <= sum(vector_of_daily_minimum_arbitrage))
    @constraint(model, [t = 1:T], ε[t, end] - ε[t, end - 1] == 0.0)
    @objective(model, Min, sum(Γ.^2))
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL
    return objective_value(model)
end

function sub_problem_smooth_through_time(args_structural_expression::Dict{Symbol,Any}, 
                                         input::StructuralModelInputs, 
                                         vector_of_daily_minimum_arbitrage::Vector{Float64})
    println(stdout, "Getting the smoothing in time weight.")
    T = length(input.vec_contracts_by_day)
    maximum_maturity_contracts = maximum_maturity_from_vec_contracts_by_day(input.vec_contracts_by_day)
    maximum_maturity = max(maximum_maturity_contracts, input.parameters["minimum_maturity"])
    
    model = Model(QUAD_SOLVER)
    # structural error
    @variable(model, ε[0:T, 0:maximum_maturity])
    # smooth through time
    @variable(model, Λ[2:T, 1:maximum_maturity])
    arbitrage_constraint = third_level_common_constraints!(model, input, args_structural_expression)
    @constraint(model, [t = 2:T - 1, j = 1:maximum_maturity], Λ[t,j] == ε[t + 1, j] - 2 * ε[t, j] + ε[t - 1, j])
    @constraint(model, arbitrage_constraint <= sum(vector_of_daily_minimum_arbitrage))
    @constraint(model, [t = 1:T], ε[t, end] - ε[t, end - 1] == 0.0)
    @objective(model, Min, sum(Λ.^2))
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL
    return objective_value(model)
end

function handle_structural_error_value(value_ε)
    ## since we created a f0, the structural error must be treated
    lin, col = size(value_ε)
    structural_error = Array{Float64}(undef, lin - 1, col - 1)
    for l in 1:lin - 1, c in 1:col - 1
        structural_error[l, c] = value_ε[l, c]
    end
    return structural_error
end

function estimate_structural_error(args_structural_expression::Dict{Symbol,Any}, 
                                   input::StructuralModelInputs,
                                   vector_of_daily_minimum_arbitrage::Vector{Float64},
                                   time_weigth::Float64, 
                                   maturity_weigth::Float64)
    println(stdout, "Estimating the structural model error.")
    T = length(input.vec_contracts_by_day)
    maximum_maturity_contracts = maximum_maturity_from_vec_contracts_by_day(input.vec_contracts_by_day)
    maximum_maturity = max(maximum_maturity_contracts, input.parameters["minimum_maturity"])
    
    model = Model(QUAD_SOLVER)
    # structural error
    @variable(model, ε[0:T, 0:maximum_maturity])
    # smooth through maturity
    @variable(model, Γ[1:T, 1:maximum_maturity])
    # smooth through time
    @variable(model, Λ[2:T, 1:maximum_maturity])
    arbitrage_constraint = third_level_common_constraints!(model, input, args_structural_expression)
    @constraint(model, [t = 1:T, j = 1:maximum_maturity - 1], Γ[t,j] == ε[t, j + 1] - 2 * ε[t, j] + ε[t, j - 1])
    @constraint(model, [t = 2:T - 1, j = 1:maximum_maturity], Λ[t,j] == ε[t + 1, j] - 2 * ε[t, j] + ε[t - 1, j])
    @constraint(model, [t = 1:T], ε[t, end] - ε[t, end - 1] == 0.0)
    @constraint(model, arbitrage_constraint <= sum(vector_of_daily_minimum_arbitrage))
    @objective(model, Min, (input.parameters["λ"] / time_weigth) * sum(Λ.^2) + ((1 - input.parameters["λ"]) / maturity_weigth) * sum(Γ.^2))
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL
    ε_val = handle_structural_error_value(JuMP.value.(ε))
    return objective_value(model), ε_val
end

function third_level_estimate_structural_error(args_structural_expression, 
                                               input, 
                                               vector_of_daily_minimum_arbitrage::Vector{Float64})
    maturity_smooth_time = @elapsed begin
        maturity_weigth = sub_problem_smooth_through_maturiy(args_structural_expression, 
                                                         input, 
                                                         vector_of_daily_minimum_arbitrage)
    end
    println(stdout, "Finished in $(round(maturity_smooth_time, digits=2)) s")

    time_smooth_time = @elapsed begin
        time_weigth = sub_problem_smooth_through_time(args_structural_expression, 
                                                         input, 
                                                         vector_of_daily_minimum_arbitrage)
    end
    println(stdout, "Finished in $(round(time_smooth_time, digits=2)) s")

    estimate_strutural_error_time = @elapsed begin
        (struct_err_obj_function, 
            structural_error) = estimate_structural_error(args_structural_expression, 
                                                        input, 
                                                        vector_of_daily_minimum_arbitrage, 
                                                        time_weigth, maturity_weigth)
    end
    println(stdout, "Finished in $(round(estimate_strutural_error_time, digits=2)) s")

    return (maturity_weigth, time_weigth, struct_err_obj_function, structural_error, 
            ThirdLevelSolveTime(maturity_smooth_time,
                                    time_smooth_time,
                                    estimate_strutural_error_time))
end

function retrieve_elementary_forward_contracts(args_structural_expression, structural_error, input)
    T, J = size(structural_error)
    X = input.X
    elementary_forward_contracts = Array{Float64}(undef, T, J)
    for t in 1:T, j in 1:J
        elementary_forward_contracts[t,j] = (structural_expression(args_structural_expression, 
                                                          X[t, j, :])
                                                            + 
                                                          structural_error[t,j]
                                                          )
    end
    return elementary_forward_contracts
end

function reconstruct_forward_contracts(elementary_forward_contracts::Matrix{Float64}, 
                                       input::StructuralModelInputs)
    

    reconstructed_vec_contracts_by_day = deepcopy(input.vec_contracts_by_day)
    num_of_contracts = number_of_contracts(reconstructed_vec_contracts_by_day)
    for (t, contracts_by_day) in enumerate(reconstructed_vec_contracts_by_day)
        for i = 1:num_of_contracts
            contract_info = contracts_by_day.contracts[i]
            maturity_interval = contract_info.time_to_maturity:contract_info.time_to_maturity + contract_info.duration - 1
            reconstructed_price = sum(elementary_forward_contracts[t, j] 
                            * (1 + input.parameters["interest_yield"])^(-j)
                            for j in maturity_interval) / calc_J(input.parameters["interest_yield"], maturity_interval)
            reconstructed_vec_contracts_by_day[t].contracts[i].price = reconstructed_price
        end
    end

    return reconstructed_vec_contracts_by_day
end

function create_ground_level_sub_problem_variables!(model::JuMP.Model, 
                                                    maximum_maturity::Int,
                                                    N::Int)
    @variable(model, ε[1:maximum_maturity])
    @variable(model, ζ[1:N])
    @variable(model, s[1:N] == 0)
    @variable(model, Δ[1:N]) 
    @variable(model, θ)
    return
end

function create_ground_level_sub_problem_constraints!(model::JuMP.Model, 
                                                      contracts_by_day::ContractsByDay, 
                                                      N::Int, 
                                                      interest_yield::Number,
                                                      index_of_traded_contracts::Vector{Int})
    for i = 1:N
        contract_info              = contracts_by_day.contracts[index_of_traded_contracts[i]]
        price                      = contract_info.price
        contract_maturity_interval = contract_info.time_to_maturity:contract_info.time_to_maturity + contract_info.duration - 1
        
        @constraint(model, contract_info.price == (
            sum(
                (model[:ε][j]) * 
                (1 + interest_yield)^(-j) for j in contract_maturity_interval
            ) / 
            calc_J(interest_yield, contract_maturity_interval)
            ) + 
                model[:ζ][i]
            )
        @constraint(model, model[:Δ][i] >= model[:ζ][i] + model[:s][i])
        @constraint(model, model[:Δ][i] >= -model[:ζ][i] + model[:s][i])
    end
    @constraint(model, infinity_norm[i=1:N], model[:θ] >= model[:Δ][i])
    return
end

function create_ground_level_sub_problem_objective!(model::JuMP.Model)        
    return @objective(model, Min, model[:θ])
end

function optimize_ground_level(model::JuMP.Model, N::Int)
    optimize!(model)
    if (termination_status(model) != MOI.OPTIMAL)
        error("O problema não foi resolvido até a otimalidade. $(termination_status(model))")
    end
    minimum_arbitrage_by_contract = Vector{Float64}(undef, N)
    while termination_status(model) == MOI.OPTIMAL
        _, i_max = findmax(JuMP.dual.(model[:infinity_norm]))
        minimum_arbitrage_by_contract[i_max] = value.(model[:ζ])[i_max]
        fix(model[:ζ][i_max], minimum_arbitrage_by_contract[i_max])
        unfix(model[:s][i_max])
        optimize!(model)
    end
    return minimum_arbitrage_by_contract
end

function sub_problem_contracts_minimum_arbitrage(contracts_by_day,
                                                  interest_yield)
    maximum_maturity = maximum_maturity_from_contracts_by_day(contracts_by_day)
    index_of_traded_contracts, N = get_index_and_length_of_traded_contracts(contracts_by_day)

    if N != 0
        model = Model(LIN_SOLVER)
        create_ground_level_sub_problem_variables!(model, maximum_maturity, N)
        create_ground_level_sub_problem_constraints!(model, contracts_by_day, N, 
                                                     interest_yield,
                                                     index_of_traded_contracts)
        create_ground_level_sub_problem_objective!(model)
        
        minimum_arbitrage_by_contract = optimize_ground_level(model, N)

        return minimum_arbitrage_by_contract
    else
        return []
    end
end

function create_arbitrage_free_input(input::StructuralModelInputs, 
                                     vector_of_minimum_arbitrage_by_contract::Vector{Vector{Float64}})
    
    arbitrage_free_input = deepcopy(input)
    vec_contracts_by_day = input.vec_contracts_by_day 
    
    for (t, contracts_by_day) in enumerate(vec_contracts_by_day)
        j = 1
        for (i, contracts_info) in enumerate(contracts_by_day.contracts)
            if !isnan(contracts_info.price)
                arbitrage_free_price = (
                        contracts_info.price - vector_of_minimum_arbitrage_by_contract[t][j]
                )
                arbitrage_free_input.vec_contracts_by_day[t].contracts[i].price = arbitrage_free_price
                j += 1
            end
        end
    end
    
    return arbitrage_free_input
end

function ground_level_arbitrage_free_input(input::StructuralModelInputs)
    ### criar problema de otimização para recuperar a arbitragem de cada contratos
    println(stdout, "Creating arbitrage-free forward contracts.")
    vector_of_minimum_arbitrage_by_contract = Vector{Vector{Float64}}(undef, length(input.vec_contracts_by_day))
    for (t, contracts_by_day) in enumerate(input.vec_contracts_by_day)
        minimum_arbitrage_by_contract = sub_problem_contracts_minimum_arbitrage(
                                                contracts_by_day,
                                                input.parameters["interest_yield"]
                                            )
        vector_of_minimum_arbitrage_by_contract[t] = minimum_arbitrage_by_contract
    end

    return (create_arbitrage_free_input(input, vector_of_minimum_arbitrage_by_contract), 
        vector_of_minimum_arbitrage_by_contract)
end

function forward_structural_model(input::StructuralModelInputs)
    ground_level_time = @elapsed begin
        (arbitrage_free_input, 
            vector_minimum_arbitrage) = ground_level_arbitrage_free_input(input)
    end
    println(stdout, "Finished in $(round(ground_level_time, digits=2)) s")

    first_level_time = @elapsed begin
        (struct_AdaLASSO, 
            select_variables_idx, 
              args_structural_expression, 
                glm_model) = first_level_estimate_parameters(arbitrage_free_input)
        
        arbitrage_free_input.X = arbitrage_free_input.X[:,:,select_variables_idx] # removing non selected variables by adaLasso
    end
    println(stdout, "Finished in $(round(first_level_time, digits=2)) s")

    second_level_time = @elapsed begin
        vector_of_daily_minimum_arbitrage = second_level_daily_minimum_arbitrage(
                                                args_structural_expression, 
                                                arbitrage_free_input
                                            )
    end
    println(stdout, "Finished in $(round(second_level_time, digits=2)) s")

    (maturity_weigth, time_weigth, 
        struct_err_obj_function, structural_error,
            third_level_time) = third_level_estimate_structural_error(
                                                args_structural_expression, 
                                                arbitrage_free_input, 
                                                vector_of_daily_minimum_arbitrage)

    elementary_forward_contracts = retrieve_elementary_forward_contracts(
                                                args_structural_expression, 
                                                structural_error, arbitrage_free_input)

    structural_dinamic = elementary_forward_contracts .- structural_error

    reconstructed_forward_prices = reconstruct_forward_contracts(
                                                elementary_forward_contracts, 
                                                arbitrage_free_input)

    return StructuralModelOutputs(
                glm_model,
                args_structural_expression,
                vector_of_daily_minimum_arbitrage,
                maturity_weigth,
                time_weigth,
                struct_err_obj_function,
                structural_error,
                structural_dinamic,
                reconstructed_forward_prices,
                StructuralSolveTime(
                        ground_level_time,
                        first_level_time,
                        second_level_time,
                        third_level_time
                    ),
                struct_AdaLASSO,
                select_variables_idx,
                vector_minimum_arbitrage
                )
end
