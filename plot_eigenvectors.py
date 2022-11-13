import environment
import options

# Environment we wish to find eigenvectors for
opt_env = environment.GridEnvironment()
opt = options.Options(opt_env, alpha=0.1, epsilon=1.0, discount=0.9)
opt.display_eigenvector(env=opt_env)