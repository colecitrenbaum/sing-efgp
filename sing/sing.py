import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from sing.initialization import initialize_params
from sing.inputs import InputSignals
from sing.likelihoods import Likelihood
from sing.sde import SDE
from sing.utils.sing_helpers import *
from sing.utils.params import *
from sing.utils.general_helpers import sgd

from typing import Any, Dict, Optional, Tuple

def _symm(M): # be sure to symmetrize gradients wrt E[xxT]
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))

def nat_grad_likelihood(mean_params: MeanParams, t_mask: jnp.array, ys_obs: jnp.array, likelihood: Likelihood, output_params: Dict[str, jnp.array]) -> NaturalParams:
    """
    Computes the natural gradient of the expected log likelihood under the current variational posterior,
    E_{q(x_i)}[log p(y_i| x_i)] delta(tau_i) , i = 0, 1, ..., T
    where delta(tau_i) is =1 if an observation occurs at time tau_i and is =0 if not

    Used for computing the SING updates
    """

    def _compute_grads(mean_params, y):
        all_grad_mu1, all_grad_mu2 = vmap(partial(likelihood.grad_ell, mean_params))(y, output_params) # vmap over obs dims
        grad_mu1, grad_mu2 = all_grad_mu1.sum(0), all_grad_mu2.sum(0)

        nat_grad = {'J': grad_mu2, 'h': grad_mu1}
        return nat_grad

    mean_params = {'Ex': mean_params['Ex'], 'ExxT': mean_params['ExxT']}
    nat_grad = vmap(_compute_grads)(mean_params, ys_obs) # vmap over observation times t_i
    
    return {
        'J': jnp.where(t_mask[:, None, None], _symm(nat_grad['J']), jnp.zeros_like(nat_grad['J'])),
        'h': jnp.where(t_mask[:, None], nat_grad['h'], jnp.zeros_like(nat_grad['h']))
    }

def nat_grad_transition(key: jr.PRNGKey, fn: SDE, gp_post: GPPost, drift_params: Dict[str, Any], trial_mask: jnp.array, init_params: InitParams, t_grid: jnp.array, mean_params: MeanParams, inputs: jnp.array, input_effect: jnp.array, sigma: float) -> NaturalParams:
    """
    Computes the natural gradient of the expected log prior transition probabilities under the current variational posterior,
    E_{q(x_i, x_{i+1})}[log p(x_{i+1}| x_i)], i = 0, ..., T-1

    Used for computing the SING updates
    """
    def _compute_init_neg_CE_wrapped(mean_params_init):
        """
        Computes the initial negative cross entropy E_{q(x_0)}[log p(x_0)] with respect to mean params
        """
        Ex, ExxT = mean_params_init['Ex'], mean_params_init['ExxT']
        m0, S0 = Ex, ExxT - jnp.outer(Ex, Ex) # convert mean -> marginal params
        return compute_neg_CE_initial(m0, S0, init_params['mu0'], init_params['V0'])

    def _compute_neg_CE_wrapped(key, mean_params, EEx, EExxT, input_t, t, del_t):
        """
        Computes the negative cross entropy E_{q(x_i, x_{i+1})}[log p(x_{i+1}| x_i)] for i = 0,...,T-1 with respect to mean params
        """
        # First perform the change-of-variable (\mu_i, \mu_{i+1}) -> (m_i, S_i, S_{i+1, i}, m_{i+1}, S_{i+1}, S_{i+2, i+1} + m_{i+2}m_{i+1}^T)
        mean_params = {'Ex': jnp.stack([mean_params['Ex'], EEx], axis=0), # (2, D)
        'ExxT': jnp.stack([mean_params['ExxT'], EExxT], axis=0), # (2, D, D)
        'ExxnT': jnp.expand_dims(mean_params['ExxnT'], 0) # (1, D, D)
        }
        marginal_params = mean_to_marginal_params(mean_params) # (2, D), (2, D, D), (1, D, D)
        m, S, SS = marginal_params['m'], marginal_params['S'], marginal_params['SS']

        # Then compute neg CE on this pair of time points
        return compute_neg_CE_single(fn, gp_post, drift_params, key, t, del_t, m[0], m[1], S[0], S[1], SS[0], input_t, input_effect, sigma)

    # Do not need the marginal mean, covariance at the last step
    D, T = mean_params['Ex'].shape[-1], t_grid.shape[0]
    mean_params_init = {
        'Ex': mean_params['Ex'][0], # (D)
        'ExxT': mean_params['ExxT'][0] # (D, D)
    }
    mean_params_modified = {
        'Ex': mean_params['Ex'][:-1], # (T-1, D)
        'ExxT': mean_params['ExxT'][:-1], # (T-1, D, D)
        'ExxnT': mean_params['ExxnT'] # (T-1, D, D)
    }
    # We need the means at all time points in order to convert between mean parameters and pairwise marginal parameters
    EEx = mean_params['Ex'][1:] # (T-1, D)
    EExxT = mean_params['ExxT'][1:] # (T-1, D)

    # Compute gradients 
    mean_params_grads, EEx_grads, EExxT_grads = vmap(jax.grad(_compute_neg_CE_wrapped, argnums=(1,2,3)))(jr.split(key, T-1), mean_params_modified, EEx, EExxT, inputs[:-1], t_grid[:-1], t_grid[1:] - t_grid[:-1])
    mean_params_init_grads = jax.grad(_compute_init_neg_CE_wrapped)(mean_params_init)

    trans_mask = (trial_mask[:-1] & trial_mask[1:]) # =1 iff there is a transition between x(i) and x(i+1), =0 otherwise
    nat_grad = {# first term is contribution of x(i) to p(x(i+1) | x(i)); second term is contribution of x(i) to p(x(i) | x(i-1))
        'J': jnp.concatenate([jnp.where(trans_mask[:, None, None], _symm(mean_params_grads['ExxT']), jnp.zeros((T-1, D, D))), jnp.zeros((1, D, D))], axis=0) + jnp.concatenate([jnp.zeros((1, D, D)), jnp.where(trans_mask[:, None, None], _symm(EExxT_grads), jnp.zeros((T-1, D, D)))], axis=0), # (T, D, D)
        'h': jnp.concatenate([jnp.where(trans_mask[:, None], mean_params_grads['Ex'], jnp.zeros((T-1, D))), jnp.zeros((1, D))], axis=0) + jnp.concatenate([jnp.zeros((1, D)), jnp.where(trans_mask[:, None], EEx_grads, jnp.zeros((T-1, D)))], axis=0), # (T, D)
        'L': jnp.where(trans_mask[:, None, None], mean_params_grads['ExxnT'], jnp.zeros((T-1, D, D))) # (T-1, D, D)
    }
    nat_grad['J'] = nat_grad['J'].at[0].add(jnp.where(trial_mask[0], _symm(mean_params_init_grads['ExxT']), jnp.zeros((D, D))))
    nat_grad['h'] = nat_grad['h'].at[0].add(jnp.where(trial_mask[0], mean_params_init_grads['Ex'], jnp.zeros((D))))

    return nat_grad

def sing_update(fn: SDE, likelihood: Likelihood, t_grid: jnp.array, key: jr.PRNGKey, ys: jnp.array, t_mask: jnp.array, trial_mask: jnp.array, natural_params: NaturalParams, init_params: InitParams, inputs: jnp.array, gp_post: GPPost, drift_params: Dict[str, Any], output_params: Dict[str, jnp.array], input_effect: jnp.array, sigma: float, rho: float, n_iters: int, jitter=1e-4) -> NaturalParams:
    """
    Performs a variational inference using SING

    Params
    -------------
    fn: the prior SDE
    likelihood: the likelihood p(y|x)
    t_grid: a shape (T) array, the time grid \tau on which both the continuous-time prior and variational posterior are discretized
    key: a random key, used for computing expectations under the variational posterior with Monte Carlo
    ys: a shape (batch_size, T, N) array, the observations 
    t_mask: a shape (batch_size, T) array, a binary mask indicating the presence of an observation at a given time for a given trial
    trial_mask: a shape (batch_size, T) array, a binary mask indicating whether or not a trial is active at a given time
    natural_params: the natural parameters of the variational posterior over x
    init_params: a dictionary containing the initial parameters of the prior SDE
    gp_post: for SING-GP, a dictionary containing the parameters of the variational posterior over inducing points
    drift_params: a dictionary containing the parameters of the prior SDE drift
    output_params: a dictionary containing the parameters of the model likelihood (i.e. the affine mapping C, d)
    inputs: a shape (T, n_inputs) array, the inputs to the model 
    input_effect: a shape (D, n_inputs) array, the input effect matrix
    sigma: the noise scale of the prior SDE
    rho: a scalar, the SING learning rate
    n_iters: number of SING iterations
    jitter: jitter added to the diagonal blocks of the precision during natural -> mean parameter conversion

    Returns
    -------------
    natural_params_new: a dictionary containing the new natural parameters of the variational posterior over x obtained by SING
    """

    def _perform_updates(nat_params, grads):
        """
        Updates natural parameters from a dictionary of natural gradients
        """
        return {
            'J': (1 - rho) * nat_params['J'] + rho * grads['J'],
            'L': (1 - rho) * nat_params['L'] + rho * grads['L'], 
            'h': (1 - rho) * nat_params['h'] + rho * grads['h']
        }

    def _sing_step(carry, x):
        nat_params = carry
        key = x

        # Convert natural -> mean parameters
        mean_params, _ = natural_to_mean_params(nat_params, trial_mask, jitter=jitter)

        # Compute natural gradients of the likelihood and transition terms in the ELBO
        lik_nat_grads = nat_grad_likelihood(mean_params, t_mask, ys, likelihood, output_params) # NOTE: likelihood can always be computed with 1D quadrature 
        trans_nat_grads = nat_grad_transition(key, fn, gp_post, drift_params, trial_mask, init_params, t_grid, mean_params, inputs, input_effect, sigma)
        all_grads = {
            'J': lik_nat_grads['J'] + trans_nat_grads['J'],
            'h': lik_nat_grads['h'] + trans_nat_grads['h'],
            'L': trans_nat_grads['L']
        }
        
        # Perform SING update
        nat_params_new = _perform_updates(nat_params, all_grads)

        def _perform_valid_update(ExxT, nat_params, nat_params_new):
            """
            Perform an extra check to ensure natural parameters remain valid after the natural gradient update
            """
            has_nan = jnp.any(jnp.isnan(ExxT))
            def true_branch(_):
                return nat_params
            def false_branch(_):
                return nat_params_new
            return lax.cond(has_nan, true_branch, false_branch, operand=None)
            
        # Recompute mean parameters to check for invalid natural parameters 
        mean_params_new, _ = natural_to_mean_params(nat_params_new, trial_mask, jitter=jitter)

        # Update sparse and dense parameters to new values if and only if they lie within the natural parameter space
        nat_params = _perform_valid_update(mean_params_new['ExxT'], nat_params, nat_params_new)  
        
        return nat_params, None

    # Initialize carry and iterate
    carry = natural_params
    xs = jr.split(key, n_iters)
    natural_params_final, _ = jax.lax.scan(_sing_step, carry, xs, length=n_iters)

    return natural_params_final

def compute_elbo_over_batch(key: jr.PRNGKey, ys_obs: jnp.array, t_mask: jnp.array, trial_mask: jnp.array, fn: SDE, likelihood: Likelihood, t_grid: jnp.array, drift_params: Dict[str, Any], init_params: InitParams, output_params: Dict[str, jnp.array], natural_params: NaturalParams, marginal_params: MarginalParams, Ap: float, inputs: jnp.array, input_effect: jnp.array, sigma: float) -> Tuple[float, float, float, float]:
    """
    Compute the ELBO over a batch of trials
    """
    # Perform M-step jointly through dynamics update and rest of ELBO (see Hu et al. 2024)
    batch_size = marginal_params['m'].shape[0]
    key, key_dyn = jr.split(key, 2) 
    gp_post = fn.update_dynamics_params(key_dyn, t_grid, marginal_params, trial_mask, drift_params, inputs, input_effect, sigma)

    # Compute likelihood over a batch of trials
    ell_term = vmap(partial(likelihood.ell_over_time, output_params=output_params))(ys_obs, {'m': marginal_params['m'], 'S': marginal_params['S']}, t_mask).sum()

    # Compute KL[q||p] term over a batch of trials
    entropy = vmap(compute_gaussian_entropy)(natural_params, marginal_params, Ap).sum() # vmap over trials
    neg_CE = vmap(partial(compute_neg_CE, t_grid, fn, gp_post, drift_params, input_effect=input_effect, sigma=sigma))(init_params, jr.split(key, batch_size), marginal_params, inputs, trial_mask).sum()
    KL_term = (-1) * (entropy + neg_CE)

    # Compute prior negative KL term (shared across trials in batch)
    prior_term = fn.prior_term(drift_params, gp_post)

    elbo = ell_term - KL_term + prior_term
    return elbo, ell_term, KL_term, prior_term

def fit_variational_em(key: jr.PRNGKey, 
                       fn: SDE, 
                       likelihood: Likelihood, 
                       t_grid: jnp.array, 
                       drift_params: Dict[str, Any], 
                       init_params: InitParams, 
                       output_params: Dict[str, jnp.array], 
                       trial_mask: Optional[jnp.array] = None,
                       input_signals: Optional[InputSignals] = None,
                       batch_size: Optional[int] = None, 
                       rho_sched: Optional[jnp.array] = None, 
                       sigma: float = 1.0, 
                       n_iters: int = 25, 
                       n_iters_e: int = 25, 
                       perform_m_step: bool = True, 
                       learn_output_params: bool = True,
                       n_iters_m: int = 25, 
                       learning_rate: Optional[jnp.array] = None, 
                       print_interval: int = 1) -> Tuple[MarginalParams, NaturalParams, GPPost, Dict[str, Any], InitParams, Dict[str, jnp.array], jnp.array, list[float]]:
    """
    Performs variational EM (inference and learning) with SING.

    Params:
    ------------
    key: jr.PRNGKey, random key for sampling mini-batches and Monte Carlo
    fn: the prior SDE
    likelihood: the likelihood model p(y|x)
    t_grid: a shape (T) array, the time grid \tau on which to discretize process
    drift_params: dict containing parameters of prior SDE drift (or kernel parameters if drift has Gaussian process prior) 
    init_params: dict containing the initial parameters of the prior SDE, including 
        - mu0: (D) mean of p(x0)
        - V0: (D, D) covariance of p(x0)
    output_params: dict containing parameters of likelihood model p(y|x)
    trial_mask: a shape (batch_size, T) array, a binary mask indicating whether or not a trial is active at a given time
    input_signals: InputSignals object, or None if no inputs
    batch_size: number of trials to sample per mini-batch
    rho_sched: a shape (n_iters) array with SING learning rates per iter
    sigma: prior SDE noise scale
    n_iters: number of variational EM iterations 
    n_iters_e: number of SING inference (e-steps) to run
    perform_m_step: bool, whether to perform M-step or not
    learn_output_params: bool, whether or not to update the output (likelihood) parameters
    n_iters_m: number of m-steps to run if using an optimizer (default Adam)
    learning_rate: a shape (n_iters) array with M-step learning rates per iter
    print_interval: how often to print metrics during variational EM
    NOTE: SING assumes each trial begins at time t=0

    Returns:
    ------------
    marginal_params: sufficient statistics of the variational posterior over latents x
    natural_params: natural parameters of the variational posterior over latents x
    gp_post: for SING-GP, the variational posterior over the drift f; else, None
    drift_params: learned drift parameters
    init_params: learned initial parameters
    output_params: learned likelihood parameters
    input_effect: a shape (D, n_inputs) array, the the linear map from inputs to the learned latent space
    elbos: a length (n_epochs) list containing the ELBO values following each E-step
    """
    @jit
    def _step(key, batch_args, gp_post, drift_params, output_params, input_effect, rho, learning_rate):
        # Unpack batch arguments
        ys, t_mask, trial_mask, natural_params, init_params, inputs = batch_args

        e_key, m_key, eval_key = jr.split(key, 3) # keys for e-step, m-step, and ELBO evaluation
        # E-step: Perform SING to update the natural parameters of the variational posterior on the grid \tau
        natural_params = vmap(
            partial(sing_update, fn, likelihood, t_grid, gp_post=gp_post, drift_params=drift_params, output_params=output_params, input_effect=input_effect, sigma=sigma, rho=rho, n_iters=n_iters_e)
            )(jr.split(e_key, batch_size), *batch_args)

        # Convert natural to marginal params
        marginal_params, Ap = vmap(natural_to_marginal_params)(natural_params, trial_mask) 

        dynamics_key, output_key, drift_key, input_key = jr.split(m_key, 4)
        def _m_step(m_step_args):
            gp_post, init_params, drift_params, output_params, input_effect = m_step_args

            # Update p(x0) = N(x0|mu0, V0)
            init_params = vmap(update_init_params)(marginal_params['m'][:,0], marginal_params['S'][:,0])

            # Update output params
            def _update_output_params(x):
                loss_fn_output_params = lambda key, output_params: -compute_elbo_over_batch(key, ys, t_mask, trial_mask, fn, likelihood, t_grid, drift_params, init_params, output_params, natural_params, marginal_params, Ap, inputs, input_effect, sigma)[0]
                output_params = likelihood.update_output_params(marginal_params, key=output_key, output_params=x, ys=ys, t_mask=t_mask, loss_fn=loss_fn_output_params, n_iters_m=n_iters_m)
                return output_params
            output_params = lax.cond(learn_output_params, _update_output_params, lambda args: args, output_params) 
            
            # Update drift params
            loss_fn_drift_params = lambda key, drift_params: -compute_elbo_over_batch(key, ys, t_mask, trial_mask, fn, likelihood, t_grid, drift_params, init_params, output_params, natural_params, marginal_params, Ap, inputs, input_effect, sigma)[0]
            drift_params, _ = sgd(drift_key, loss_fn_drift_params, drift_params, n_iters=n_iters_m, learning_rate=learning_rate)

            # Update input effect matrix B
            input_effect = input_signals.update_input_effect(input_key, fn, t_grid, marginal_params, trial_mask, inputs, gp_post, drift_params)

            return gp_post, init_params, drift_params, output_params, input_effect
        
        # M-step: Update the likelihood (output) parameters, prior drift parameters, and input effect matrix
        m_step_args = (gp_post, init_params, drift_params, output_params, input_effect)
        _, init_params, drift_params, output_params, input_effect = lax.cond(perform_m_step, _m_step, lambda args: args, m_step_args)

        # Update variational posterior on GP drift if fn is a SparseGP, else set to None
        gp_post = fn.update_dynamics_params(dynamics_key, t_grid, marginal_params, trial_mask, drift_params, inputs, input_effect, sigma)

        # Compute the ELBO at end of the vEM iteration on the batch
        elbo_terms = compute_elbo_over_batch(eval_key, ys, t_mask, trial_mask, fn, likelihood, t_grid, drift_params, init_params, output_params, natural_params, marginal_params, Ap, inputs, input_effect, sigma)

        return natural_params, marginal_params, gp_post, drift_params, init_params, output_params, input_effect, elbo_terms

    n_trials, n_timesteps, _ = likelihood.ys_obs.shape
    latent_dim = fn.latent_dim

    # Initialize rho schedule for SING updates
    if rho_sched is None:
        rho_sched = jnp.logspace(-3, -1, num=n_iters)
    else:
        assert rho_sched.shape[0] == n_iters

    # Initialize learning rate schedule for M-step
    if learning_rate is None:
        learning_rate = jnp.arange(1, n_iters+1) ** (-0.5)
    else:
        assert learning_rate.shape[0] == n_iters
    
    # Check for inputs
    if input_signals is None:
        v = jnp.zeros((n_trials, n_timesteps, 1))
        input_signals = InputSignals(v)
    input_effect = jnp.zeros((fn.latent_dim, input_signals.v.shape[-1])) # linear mapping of the inputs into the latent

    # If no batch size if given, use all trials
    if batch_size is None:
        batch_size = n_trials

    if learn_output_params is True: assert perform_m_step is True

    # If no trial_mask, all trials have same length 
    if trial_mask is None:
        trial_mask = jnp.ones((n_trials, n_timesteps))
    else:
        assert trial_mask.shape == (n_trials, n_timesteps)
    trial_mask = trial_mask.astype(bool)
    
    # Initialize natural parameters
    key, init_key = jr.split(key, 2)
    print("Initializing params...")
    natural_params, marginal_params, gp_post = vmap(
        partial(initialize_params, fn, drift_params, t_grid=t_grid, sigma=sigma), out_axes=(0, 0, None)
        )(jr.split(init_key, n_trials), init_params, trial_mask)

    elbos = []
    
    print("Performing variational EM algorithm...")
    for i in range(n_iters):
        # Keys for mini-batching and sampling
        key_batch, key_sample = jr.fold_in(key, 2*i), jr.fold_in(key, 2*i + 1)

        batch_inds = jr.choice(key_batch, n_trials, (batch_size,), replace=False)

        # Subset batches
        batch_args = subset_batches([likelihood.ys_obs, likelihood.t_mask, trial_mask, natural_params, init_params, input_signals.v], batch_inds)

        # Perform a vEM step  
        natural_params_batch, marginal_params_batch, gp_post, drift_params, init_params_batch, output_params, input_effect, elbo_terms = _step(key_sample, batch_args, gp_post, drift_params, output_params, input_effect, rho_sched[i], learning_rate[i])
        
        # Fill batches
        natural_params, marginal_params, init_params = fill_batches([natural_params, marginal_params, init_params], [natural_params_batch, marginal_params_batch, init_params_batch], batch_inds)

        # Log ELBO terms
        elbo, ell_term, KL_term, prior_term = elbo_terms
        if (i + 1) % print_interval == 0:
            print(f"Iteration {i + 1} / {n_iters}, ELBO: {elbo}, ell: {ell_term}, KL: {KL_term}, prior: {prior_term}")
        elbos.append(elbo)

    return marginal_params, natural_params, gp_post, drift_params, init_params, output_params, input_effect, elbos