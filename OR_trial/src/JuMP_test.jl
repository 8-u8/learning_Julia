module OR_trial
# from: https://jump.dev/JuMP.jl/stable/tutorials/linear/diet/
# 食事問題を解く。
using JuMP
import DataFrames
import HiGHS

foods = DataFrames.DataFrame(
    [
        "hamburger" 2.49 410 24 26 730
        "chicken" 2.89 420 32 10 1190
        "hot dog" 1.50 560 20 32 1800
        "fries" 1.89 380 4 19 270
        "macaroni" 2.09 320 12 10 930
        "pizza" 1.99 320 15 12 820
        "salad" 2.49 320 31 12 1230
        "milk" 0.89 100 8 2.5 125
        "ice cream" 1.59 330 8 10 180
    ],
    ["name", "cost", "calories", "protein", "fat", "sodium"],
)

limits = DataFrames.DataFrame(
    [
        "calories" 1800 2200
        "protein" 91 Inf
        "fat" 0 65
        "sodium" 0 1779
    ],
    ["nutrient", "min", "max"],
)

model = Model(HiGHS.Optimizer)
set_silent(model)


@variable(model, x[foods.name]>=0)

foods.x = Array(x)

@objective(model, Min, sum(foods.cost .* foods.x)); # .* は何？

@constraint(
    model,
    [row in eachrow(limits)],
    row.min <= sum(foods[!, row.nutrient] .* foods.x) <= row.max,

);

print(model)

optimize!(model)
solution_summary(model)

for row in eachrow(foods)
    println(row.name, "=", value(row.x))
end

end 