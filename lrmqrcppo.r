## Tests for lrmqrcppo.stan partial proportional odds model

require(rms)
require(rstan)
mod <- stan_model('lrmqrcppo.stan')

## Test 1: PO model (partial PPO terms heavily penalized)

p0 <- c(.4, .2, .4)
p1 <- c(.3, .1, .6)
m  <- 50
m0 <- p0 * m
m1 <- p1 * m
x  <- c(rep(0, m), rep(1, m))
y0 <- c(rep(1, m0[1]), rep(2, m0[2]), rep(3, m0[3]))
y1 <- c(rep(1, m1[1]), rep(2, m1[2]), rep(3, m1[3]))
y  <- c(y0, y1)
table(x, y)
xc <- x - mean(x)

f <- lrm(y ~ xc)
f
predict(f, data.frame(xc=c(-0.5, 0.5)), type='fitted.ind')

require(VGAM)
fv <- vgam(y ~ xc, cumulative(reverse=TRUE, parallel=TRUE))
coef(fv)
predict(fv, data.frame(xc=c(-.5,.5)), type='response')
fvppo <- vgam(y ~ xc, cumulative(reverse=TRUE, parallel=FALSE))
co <- coppo <- coef(fvppo)
co
co['xc:2'] - co['xc:1']
predict(fvppo, data.frame(xc=c(-.5, .5)), type='response')
pge2 <- plogis(co[1] + c(-0.5, 0.5) * co[3])
peq3 <- plogis(co[2] + c(-0.5, 0.5) * co[4])
rbind(c(1 - pge2[1], pge2[1] - peq3[1], peq3[1]),
      c(1 - pge2[2], pge2[2] - peq3[2], peq3[2]))

d <- list(N=length(x), Nc=length(x), p=1, q=1,
          X=cbind(xc), Z=cbind(xc), y=y, k=3,
          cluster=1 : length(x),
          sds=as.array(1000),  sdsppo=as.array(0.001), # didn't work: .0001
          rate=10000000.)
opt <- function(r=1000) {
  b <- rep(0, 5)
  for(i in 1 : r) {
    b <- b + optimizing(mod, data=d)$par[nam]
  }
  b / r
}
nam <- c('sigmag', 'alpha[1]', 'alpha[2]', 'beta[1]', 'tau[1,1]')
cmine <- opt()
cmine

clogdiff <- opt()
clogdiff
coef(f)

# Now relax the prior on tau
d$sdsppo <- as.array(1000)
gppo <- optimizing(mod, data=d)
gppo$par[nam]
gppo$par['omega[1,1]']
co
co['xc:2'] - co['xc:1']

co <- gppo$par[nam][-1]
pge2 <- plogis(co[1] + c(-0.5, 0.5) * co[3])
peq3 <- plogis(co[2] + c(-0.5, 0.5) * (co[3] + co[4]))
rbind(c(1 - pge2[1], pge2[1] - peq3[1], peq3[1]),
      c(1 - pge2[2], pge2[2] - peq3[2], peq3[2]))

gppo <- sampling(mod, data=d)
summary(gppo, par=nam)

# Now try blrm
b <- blrm(y ~ x, ~x, priorsd=1000, priorsdppo=1000, method='opt')
b <- blrm(y ~ x, ~ x, priorsd=1000, priorsdppo=1000)
b
apply(b$draws, 2, mean)

# Get vgam ppo estimates on original non-centered x scale
fvppo <- vgam(y ~ x, cumulative(reverse=TRUE, parallel=FALSE))
coef(fvppo)
# take differences in last 2 coefficients to get our scheme


## Check Stan calculation
a <- 1; b <- 0.8; log(plogis(a) - plogis(b))
log(1 / (1 + exp(-a)) - 1 / (1 + exp(-b)))
log1p_exp <- function(x) log(1 + exp(x))
log_diff_exp <- function(x, y) log(exp(x) - exp(y))
log_diff_exp(-log1p_exp(-a), -log1p_exp(-b))
