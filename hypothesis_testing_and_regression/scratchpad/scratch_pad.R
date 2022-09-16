dbinom(63, size = 31000, prob = 2.03/1000)

#Fisher exact test
#contingency table 

deathTC = matrix(c(39, 63, 30961, 30937), nrow=2, ncol=2, dimnames = list(Treatment=c("Death", "Survival"), Control=c("Death", "Survival")))

fisher.test(deathTC)

#z-test
pnorm(-3.0268)

