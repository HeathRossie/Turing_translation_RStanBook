# StanとRでベイズ統計モデリングのJulia-Turingバージョン
# chapter 07
# written by Hiroshi Matsui

using Turing, StatsPlots, Distributions, Plots, CSV, DataFrames

### model 7-3 (1,2は省略)
# 非線形回帰
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-aircon.txt") |> CSV.File |> DataFrame

@model function mod73(X, Y)
    a ~ Normal(0, 100)
    b ~ Normal(0, 100)
    x0 ~ truncated(Normal(0, 100), 0, Inf)
    σ ~ truncated(Normal(0, 100), 0, Inf)
    for i in 1:length(Y)
        Y[i] ~ Normal(a + b*(X[i]-x0)^2, σ)
    end
end

chn = sample(mod73(d.X, d.Y), NUTS(), 2000)



### model 7-4 
# 非線形回帰
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-conc.txt") |> CSV.File |> DataFrame
plot(d.Time, d.Y, legend=false)
scatter!(d.Time, d.Y, legend=false, color="blue")

@model function mod74(Time, Y)
    a ~ truncated(Normal(0,100),0,100)
    b ~ truncated(Normal(0,100),0,5)
    σ ~ truncated(Normal(0,100),0,Inf)
    for i in 1:length(Y)
        Y[i] ~ Normal(a*(1-exp(-b*Time[i])), σ)
    end
end

chn = sample(mod74(d.Time, d.Y), NUTS(), 2000)


### model 7-5
# 簡単なパス解析
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-50m.txt") |> CSV.File |> DataFrame

p1 = scatter(d.Age, d.Weight, legend=false)
p2 = scatter(d.Weight, d.Y, legend=false)
p3 = scatter(d.Age, d.Y, legend=false)
plot!(p1,p2,p3)

@model function mod75(Age, Weight, Y)
    
    c1 ~ Normal(0, 100)
    c2 ~ Normal(0, 100)
    b1 ~ Normal(0, 100)
    b2 ~ Normal(0, 100)
    b3 ~ Normal(0, 100)
    σ_w ~ truncated(Normal(0, 100), 0, Inf)
    σ_Y ~ truncated(Normal(0, 100), 0, Inf)

    for i in 1:length(Age)
        Weight[i] ~ Normal(c1 + c2*Age[i], σ_w)
        Y[i] ~ Normal(b1 + b2*Age[i] + b3*Weight[i], σ_Y)
    end
end

chn = sample(mod75(d.Age, d.Weight, d.Y), NUTS(), 2000)
chn = chn[1000:2000]

### model 7-5
# 説明変数にノイズが含まれる場合
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap04/input/data-salary.txt") |> CSV.File |> DataFrame

@model function mod76(X, Y)
    a ~ Normal(0, 100)   
    b ~ Normal(0, 100)   
    σ_Y ~ truncated(Normal(0,100), 0, Inf)
    x_true ~ Normal(0,100)


    for i in 1:length(Y)
        X[i] ~ Normal(x_true, 2.5)
        Y[i] ~ Normal(a + b*x_true, σ_Y)
    end
end

chn = sample(mod76(d.X, d.Y), NUTS(), 2000)
chn = chn[1000:2000]


### model 7-7
# データに打ち切りがあるケース
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-protein.txt") |> CSV.File |> DataFrame
d.Y2 = copy(d.Y)
d.censor = repeat([0.], length(d.Y))
d[d.Y.=="<25",:censor] .= 1
d[d.Y.=="<25",:Y2] .= "0"
d.Y2 = parse.(Float64, d.Y2)


@model function mod77(Y, censor)
    µ ~ Normal(0, 2)
    σ ~ truncated(Normal(0, 100), 0, Inf)
    for i in 1:length(Y)
        if censor[i] == 0  # observed
            Turing.@addlogprob! logpdf(Normal(µ, σ), Y[i])
        elseif censor[i] == 1  # censored
            Turing.@addlogprob! logccdf(Normal(µ, σ), Y[i])
        end
    end
end

chn = sample(mod77(d.Y2, d.censor), NUTS(), 2000)
chn = chn[1000:2000]


### model 7-9 
# 外れ値
d = download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-outlier.txt") |> CSV.File |> DataFrame
scatter(d.X, d.Y, legend=false)

@model function mod79(X, Y)
    a ~ Normal(0, 100)
    b ~ Normal(0, 100)
    σ ~ truncated(Normal(0, 100), 0, Inf)
    for i in 1:length(Y)
        Y[i] ~ Cauchy(a + b*X[i], σ)
    end
end

chn = sample(mod79(d.X, d.Y), NUTS(), 2000)
chn = chn[1000:2000]
