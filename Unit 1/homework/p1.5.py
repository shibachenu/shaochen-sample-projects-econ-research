#Publication bias discussions

alpha = 0.05
R = 0.2/0.8
prestudy_p = R/(R+1)

pwr = 0.7
beta = 1-pwr

rho = (R*pwr)/(R+alpha)
ppv = (R*pwr)/(R*pwr + alpha)

print("Prestudy prob. ", prestudy_p, "poststudy prob. /ppv: ", ppv, "rho: ", rho)