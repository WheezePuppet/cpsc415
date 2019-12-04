
import numpy as np

PD0 = np.array([.8,0,.2])

PDt = np.array([[.5,.5, 0],
                [.2,.5,.3],
                [ 0,.6,.4]])

Pf = np.array([.05,.25,.2])
Pg = np.array([.1,.3,.1])

prior_D1 = PD0.dot(PDt)
print("Prior probs of D1: {}.".format(prior_D1.round(2)))
posterior_D1 = prior_D1 * Pf * (1-Pg)  # Assuming f1 ^ ~g1
posterior_D1 = posterior_D1 / sum(posterior_D1)  # Normalize, Jake-style
print("Posterior probs of D1: {}.".format(posterior_D1.round(2)))

prior_D2 = posterior_D1.dot(PDt)
print("Prior probs of D2: {}.".format(prior_D2.round(2)))
posterior_D2 = prior_D2 * (1-Pf) * (1-Pg)  # Assuming ~f2 ^ ~g2
posterior_D2 = posterior_D2 / sum(posterior_D2)  # Normalize, Jake-style
print("Posterior probs of D2: {}.".format(posterior_D2.round(2)))
