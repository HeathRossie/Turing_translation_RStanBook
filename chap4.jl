# StanとRでベイズ統計モデリングのJulia-Turingバージョン
# chapter 04
# written by Hiroshi Matsui

using Turing, StatsPlots, Distributions, Plots, CSV, DataFrames

### model 4-1
# 分散が既知の正規分布
Y = rand(Normal(5, 1), 100)
histogram(Y)

# モデルの定義
@model function mod41(Y)
    µ ~ Normal(0, 100)
    Y ~ Normal(µ, 1)
end

# サンプリング
chn = sample(mod41(Y), NUTS(), 2000)
describe(chn)
plot(chn)


### model 4-2 ~ 4-5 (等価な表現で同じモデル)
# 単回帰分析
d = CSV.File(download("https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap04/input/data-salary.txt"))
d = DataFrame(d)
scatter(d.X, d.Y, legend=false)

# モデルの定義
@model function mod42(X, Y)
    a ~ Normal(0,100)
    b ~ Normal(0, 100)
    σ ~ truncated(Normal(0, 100), 0, Inf)
    
    for i in 1:length(Y) 
        Y[i] ~ Normal(a + b * X[i], σ)
    end
end

# サンプリング
chn = sample(mod42(d.X, d.Y), NUTS(), 2000)
chn = chn[1000:2000]

describe(chn)[1]
describe(chn)[2]
plot(chn)

### 事後予測チェック
X_new = collect(range(minimum(d.X), maximum(d.X), length=100))
function prediction(chn, X_new)
    p = get_params(chn)
    targets = p.a' .+ X_new * reduce(hcat, p.b)
    
    postpred = zeros(size(targets))
    for row in 1:size(targets)[1]
        
        for j in 1:length(p.σ)
            postpred[row,j] = rand(Normal(targets[row,j], p.σ[j]), 1)[1] 
        end

  
        
    end

    res = DataFrame(postmean = mean.(eachrow(postpred)),
                    lower = quantile.(eachrow(postpred), .025),
                    upper = quantile.(eachrow(postpred), .975),
                    X = X_new
    )
    return res
end

res = prediction(chn, X_new)

scatter(d.X, d.Y, legend=false)
plot!(res.X, res.postmean, legend=false, ribbon=(res.postmean-res.lower, res.upper - res.postmean), fillalpha=0.2)

