import attr
import numpy as np
import itertools

from attr.validators import instance_of
from numpy.random.mtrand import RandomState

from . import util

__all__ = ['Ensemble', 'EnsembleConfiguration']


@attr.s(slots=True, frozen=True)
class EnsembleConfiguration(object):
    adaptation_lag = attr.ib()
    adaptation_time = attr.ib()
    scale_factor = attr.ib()
    evaluator = attr.ib()


@attr.s(slots=True)
class Ensemble(object):
    """
    This contains as little contextual information as it can.  It represents an ensemble.py that performs steps in the
    parameter space.

    """

    _config = attr.ib(type=EnsembleConfiguration, validator=instance_of(EnsembleConfiguration))

    betas = attr.ib(type=np.ndarray, converter=util._ladder)

    # Initial walker positions and probabilities.
    x = attr.ib(type=np.ndarray, converter=np.array)
    logP = attr.ib(type=np.ndarray, default=None)
    logl = attr.ib(type=np.ndarray, default=None)

    adaptive = attr.ib(type=bool, converter=bool, default=False)

    _random = attr.ib(type=RandomState, validator=instance_of(RandomState), factory=RandomState)
    _mapper = attr.ib(default=map)
    _map_type = attr.ib(default='map')

    time = attr.ib(type=int, init=False, default=0)
    nwalkers = attr.ib(type=int, init=False)
    ntemps = attr.ib(type=int, init=False)
    ndim = attr.ib(type=int, init=False)

    jumps_proposed = attr.ib(type=np.ndarray, init=False, default=None)
    jumps_accepted = attr.ib(type=np.ndarray, init=False, default=None)
    swaps_proposed = attr.ib(type=np.ndarray, init=False, default=None)
    swaps_accepted = attr.ib(type=np.ndarray, init=False, default=None)

    # Extra proposal: branching probability and jump function
    extra_proposal_prob = attr.ib(converter=float, default=0.)
    extra_proposal_jump = attr.ib(default=None)

    # Extra tempering factor: lnL -> lnL/tempering factor
    temp_factor = attr.ib(default=1.)

    # Subset of x that is to be used for ensemble proposals
    # Allows to have ndim_ensemble ordinary parameters,
    # followed by ndim-ndim_ensemble extra params, for instance discrete
    ndim_ensemble = attr.ib(type=int, default=None)

    # Ensemble proposal:
    # ~1/sqrt(z) with volume element z^(d-1) for emcee
    # ~1/z with volume element z^d for ptemcee
    ensemble_proposal = attr.ib(converter=str, default='ptemcee')

    # Swap permutation method: random, or opposite sort (smaller val matched to largest)
    swap_perm = attr.ib(converter=str, default='random')

    # Allows to wrap parameters given a list of moduli (typically [-pi,pi]):
    # the difference between two points used for the stretch move is now the
    # 'shortest distance' difference taking into account mod.
    # NOTE: length of this must be matching ndim_ensemble
    list_param_wrap = attr.ib(default=None)

    @_mapper.validator
    def _is_callable(self, attribute, value):
        if not callable(value):
            raise ValueError('{} must be callable.'.format(attribute.name))

    @betas.validator
    def _is_consistent(self, attribute, value):
        if len(value) != len(self.x):
            raise ValueError('Number of temperatures not consistent with starting positions.')

    def __attrs_post_init__(self):
        self.ntemps, self.nwalkers, self.ndim = self.x.shape

        self.jumps_proposed = np.ones((self.ntemps, self.nwalkers))
        self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)

        # If we have no likelihood or prior values, compute them.
        if self.logP is None or self.logl is None:
            logl, logp = self._evaluate(self.x)
            self.logP = self._tempered_likelihood(logl) + logp
            self.logl = logl

        if (self.logP == -np.inf).any():
            raise ValueError('Attempting to start with samples outside posterior support.')

    def step(self):
        self._stretch(self.x, self.logP, self.logl, self.extra_proposal_prob, self.extra_proposal_jump, self.ndim_ensemble, self.list_param_wrap, self.ensemble_proposal)
        self.x = self._temperature_swaps(self.x, self.logP, self.logl, self.swap_perm)
        ratios = self.swaps_accepted / self.swaps_proposed

        # TODO: Should the notion of a 'complete' iteration really include the temperature adjustment?
        if self.adaptive and self.ntemps > 1:
            dbetas = self._get_ladder_adjustment(self.time,
                                                 self.betas,
                                                 ratios)
            self.betas += dbetas
            self.logP += self._tempered_likelihood(self.logl, betas=dbetas)

        self.time += 1

    def _stretch(self, x, logP, logl, extra_proposal_prob=0., extra_proposal_jump=None, ndim_ensemble=None, list_param_wrap=None, ensemble_proposal='ptemcee'):
        """
        Perform the stretch-move proposal on each ensemble.py.

        """

        self.jumps_accepted = np.zeros((self.ntemps, self.nwalkers))
        w = self.nwalkers // 2
        d = self.ndim
        t = self.ntemps
        loga = np.log(self._config.scale_factor)
        sqrta = np.sqrt(self._config.scale_factor)

        # Subset of dimensions to be used for ensemble proposal
        if ndim_ensemble is None:
            d_ens = d
        else:
            d_ens = ndim_ensemble

        # Presence of an extra proposal
        extra_proposal = (not extra_proposal_jump is None) and (extra_proposal_prob>0.)

        for j in [0, 1]:
            # Get positions of walkers to be updated and walker to be sampled.
            j_update = j
            j_sample = (j + 1) % 2
            x_update = x[:, j_update::2, :]
            x_sample = x[:, j_sample::2, :]

            if ensemble_proposal=='ptemcee':
                z = np.exp(self._random.uniform(low=-loga, high=loga, size=(t, w)))
                volume_exponent = d
            elif ensemble_proposal=='emcee': # Note: ~1/sqrt(z) between 1./a and a
                z = (1/sqrta + (sqrta - 1./sqrta) * self._random.uniform(low=0., high=1., size=(t, w)))**2
                volume_exponent = d-1
            else:
                raise ValueError('ensemble_proposal not recognized: %s' % ensemble_proposal)
            y = np.empty((t, w, d))
            # Case where no extra proposal is used
            if not extra_proposal:
                for k in range(t):
                    js = self._random.randint(0, high=w, size=w)
                    y[k, :, :d_ens] = util.mod_arr(x_sample[k, js, :d_ens] +
                                  z[k, :].reshape((w, 1)) *
                                  util.mod_diff_arr(x_update[k, :, :d_ens], x_sample[k, js, :d_ens], list_mod=list_param_wrap), list_mod=list_param_wrap)
                    # Keep identical dimensions beyond ndim_ensemble
                    y[k, :, d_ens:] = x_update[k, :, d_ens:]
            # Case where we use an extra proposal
            else:
                # Determining which walkers will be updated with extra proposal
                extra_p = self._random.uniform(low=0., high=1., size=(t,w))
                extra_mask = extra_p < extra_proposal_prob
                stretch_mask = ~extra_mask
                for k in range(t):
                    n_stretch = np.sum(stretch_mask[k])
                    # Update some walkers with the normal stretch move
                    js = self._random.randint(0, high=w, size=n_stretch)
                    y[k, stretch_mask[k], :d_ens] = util.mod_arr(x_sample[k, js, :d_ens] +
                                  z[k, stretch_mask[k]].reshape((n_stretch, 1)) *
                                  util.mod_diff_arr(x_update[k, stretch_mask[k], :d_ens], x_sample[k, js, :d_ens], list_mod=list_param_wrap), list_mod=list_param_wrap)
                    # Keep identical dimensions beyond ndim_ensemble
                    y[k, stretch_mask[k], d_ens:] = x_update[k, stretch_mask[k], d_ens:]
                    # Update the others with the extra proposal
                    y[k, extra_mask[k], :] = extra_proposal_jump(x_update[k, extra_mask[k], :], random_state=self._random)

            y_logl, y_logp = self._evaluate(y)
            y_logP = self._tempered_likelihood(y_logl) + y_logp

            # Acceptance probability
            logp_accept = np.zeros((t,w), dtype=float)
            # Case where no extra proposal is used
            if not extra_proposal:
                logp_accept = volume_exponent * np.log(z) + y_logP - logP[:, j_update::2]
            # Case where we use an extra proposal
            else:
                # Acceptance probability with stretch-move volume element z^d
                logp_accept[stretch_mask] = volume_exponent * np.log(z[stretch_mask]) + y_logP[stretch_mask] - logP[:, j_update::2][stretch_mask]
                # Normal acceptance probability for extra proposal
                logp_accept[extra_mask] = y_logP[extra_mask] - logP[:, j_update::2][extra_mask]

            logr = np.log(self._random.uniform(low=0, high=1, size=(t, w)))

            accepts = logr < logp_accept
            accepts = accepts.flatten()

            x_update.reshape((-1, d))[accepts, :] = y.reshape((-1, d))[accepts, :]
            logP[:, j_update::2].reshape((-1,))[accepts] = y_logP.reshape((-1,))[accepts]
            logl[:, j_update::2].reshape((-1,))[accepts] = y_logl.reshape((-1,))[accepts]

            self.jumps_accepted[:, j_update::2] = accepts.reshape((t, w))

    def _evaluate(self, x):
        """
        Evaluate the log likelihood and log prior functions at the specified walker positions.

        """

        # Make a flattened iterable of the results, of alternating logL and logp.
        shape = x.shape[:-1]
        values = x.reshape((-1, self.ndim))
        length = len(values)
        if self._map_type=='map':
            res = self._mapper(self._config.evaluator, values)
        elif self._map_type=='map_arr':
            size_send = self.ndim
            size_recv = 2
            res = self._mapper(size_recv, size_send, values) # Function is to be set when calling wait_arr of mpi pool
        results = itertools.chain.from_iterable(res)
        #results = itertools.chain.from_iterable(self._mapper(self._config.evaluator, values))

        # Construct into a pre-allocated ndarray.
        array = np.fromiter(results, float, 2 * length).reshape(shape + (2,))
        return tuple(np.rollaxis(array, -1))

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we allow the likelihood to dominate the
        temperature, since wandering outside the likelihood support causes a discontinuity.

        """

        if betas is None:
            betas = self.betas

        # NOTE: we added an extra global tempering factor applied to likelihoods
        with np.errstate(invalid='ignore'):
            loglT = logl * betas[:, None] / self.temp_factor
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        """

        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self._config.adaptation_lag / (time + self._config.adaptation_lag)
        kappa = decay / self._config.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def _temperature_swaps(self, x, logP, logl, swap_perm='random'):
        """
        Perform parallel-tempering temperature swaps on the state in ``x`` with associated ``logP`` and ``logl``.

        """

        nwalkers = self.nwalkers
        ntemps = len(self.betas)
        self.swaps_accepted = np.empty(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            if swap_perm=='random':
                iperm = self._random.permutation(nwalkers)
                i1perm = self._random.permutation(nwalkers)
            elif swap_perm=='sort_opposite':
                iperm = np.argsort(logl[i])
                i1perm = np.argsort(logl[i-1])[::-1]
            else:
                raise ValueError('swap_perm not recognized: %s' % swap_perm)

            # NOTE: we added an extra global tempering factor applied to likelihoods
            raccept = np.log(self._random.uniform(size=nwalkers))
            paccept = dbeta / self.temp_factor * (logl[i, iperm] - logl[i - 1, i1perm])

            # How many swaps were accepted?
            sel = (paccept > raccept)
            self.swaps_accepted[i - 1] = np.sum(sel)

            x_temp = np.copy(x[i, iperm[sel], :])
            logl_temp = np.copy(logl[i, iperm[sel]])
            logP_temp = np.copy(logP[i, iperm[sel]])

            x[i, iperm[sel], :] = x[i - 1, i1perm[sel], :]
            logl[i, iperm[sel]] = logl[i - 1, i1perm[sel]]
            logP[i, iperm[sel]] = logP[i - 1, i1perm[sel]] - dbeta * logl[i - 1, i1perm[sel]]

            x[i - 1, i1perm[sel], :] = x_temp
            logl[i - 1, i1perm[sel]] = logl_temp
            logP[i - 1, i1perm[sel]] = logP_temp + dbeta * logl_temp

        return x
