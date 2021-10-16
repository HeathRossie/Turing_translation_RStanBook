# StanとRでベイズ統計モデリングのJulia-Turingバージョン
# chapter 05
# written by Hiroshi Matsui

using Turing, StatsPlots, Distributions, Plots, CSV, DataFrames

### model 5-3
# 重回帰
d = CSV.File(download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-1.txt"))
d = DataFrame(d)

@model function mod53(A, Score, Y)
    b1 ~ Normal(0, 100)
    b2 ~ Normal(0, 100)
    b3 ~ Normal(0, 100)
    σ ~ truncated(Normal(0,100), 0, Inf)

    for i in 1:length(Y)
        Y[i] ~ Normal(b1 + b2 * A[i] + b3 * Score[i], σ)
    end
end

chn = sample(mod53(d.A, d.Score, d.Y), NUTS(), 2000)
chn = chn[1000:2000]
describe(chn)[1]
plot(chn)

### model 5-4 (5-5は省略)
# ロジスティック回帰
d = CSV.File(download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-2.txt"))
d = DataFrame(d)

@model function mod54(A, Score, Y, M)
    b1 ~ Normal(0, 100)
    b2 ~ Normal(0, 100)
    b3 ~ Normal(0, 100)
    for i in 1:length(Y)
        Y[i] ~ BinomialLogit(M[i], b1 + b2*A[i] + b3*Score[i])
    end
end

chn = sample(mod54(d.A, d.Score, d.Y, d.M), NUTS(), 2000)
chn = chn[1000:2000]
describe(chn)[1]
plot(chn)


### model 5-6
# ポアソン回帰
d = CSV.File(download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-2.txt"))
d = DataFrame(d)

@model function mod56(A, Score, M)
    b1 ~ Normal(0, 100)
    b2 ~ Normal(0, 100)
    b3 ~ Normal(0, 100)
    for i in 1:length(M)
        M[i] ~ Poisson(exp(b1 + b2*A[i] + b3*Score[i]))
    end
end

chn = sample(mod56(d.A, d.Score, d.Y), NUTS(), 2000)
chn = chn[1000:2000]
describe(chn)[1]
plot(chn)


