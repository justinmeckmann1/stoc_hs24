import scipy.stats as st

# k: number of successes (1) --> 1 person has the disease
# n: number of trials (100) --> 100 people
# p: probability of success on each trial (0.016) --> probability that a person has the disease

print(st.binom.cdf(k=1, n=100, p=0.016))


