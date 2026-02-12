import torch
from abc import abstractmethod
from torch.nn.utils import parameters_to_vector


def log_det_ratio(hessian, prior_prec):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def scatter(mu_q, prior_precision_diag):
    return (mu_q * prior_precision_diag) @ mu_q


def log_marginal_likelihood(mu_q, hessian, prior_prec):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = -0.5 * (
        log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec)
    )
    return neg_log_marglik


def optimize_prior_precision(mu_q, hessian, prior_prec, n_steps=100):

    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()

    return prior_prec


class BaseLaplace:
    def __init__(self):
        super(BaseLaplace, self).__init__()

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class DiagLaplace(BaseLaplace):
    def sample(self, parameters, posterior_scale, n_samples=100):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.view(1, n_params)
        return parameters.view(1, n_params) + samples

    def posterior_scale(self, hessian, scale=1, prior_prec=1):

        posterior_precision = hessian * scale + prior_prec
        return 1.0 / (posterior_precision.sqrt() + 1e-6)

    def init_hessian(self, data_size, net, device):

        hessian = data_size * torch.ones_like(parameters_to_vector(net.parameters()), device=device)
        return hessian

    def scale(self, h_s, b, data_size):
        return h_s / b * data_size

    def average_hessian_samples(self, hessian, constant):

        # average over samples
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]

        # get posterior_precision
        return constant * hessian


class BlockLaplace(BaseLaplace):
    def sample(self, parameters, posterior_scale, n_samples=100):
        n_samples = torch.tensor([n_samples])
        count = 0
        param_samples = []
        for post_scale_layer in posterior_scale:
            n_param_layer = len(post_scale_layer)

            layer_param = parameters[count : count + n_param_layer]
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                layer_param, covariance_matrix=post_scale_layer
            )
            samples = normal.sample(n_samples)
            param_samples.append(samples)

            count += n_param_layer

        param_samples = torch.cat(param_samples, dim=1).to(parameters.device)
        return param_samples

    def posterior_scale(self, hessian, scale=1, prior_prec=1):

        posterior_precision = [
            h * scale + torch.diag_embed(prior_prec * torch.ones(h.shape[0])) for h in hessian
        ]
        posterior_scale = [torch.cholesky_inverse(layer_post_prec) for layer_post_prec in posterior_precision]
        return posterior_scale

    def init_hessian(self, data_size, net, device):

        hessian = []
        for layer in net:
            # if parametric layer
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                params = parameters_to_vector(layer.parameters())
                n_params = len(params)
                hessian.append(data_size * torch.ones(n_params, n_params, device=device))

        return hessian

    def scale(self, h_s, b, data_size):
        return [h / b * data_size for h in h_s]

    def aveage_hessian_samples(self, hessian, constant):
        n_samples = len(hessian)
        n_layers = len(hessian[0])
        hessian_mean = []
        for i in range(n_layers):
            tmp = None
            for s in range(n_samples):
                if tmp is None:
                    tmp = hessian[s][i]
                else:
                    tmp += hessian[s][i]

            tmp = tmp / n_samples
            tmp = constant * tmp + torch.diag_embed(torch.ones(len(tmp), device=tmp.device))
            hessian_mean.append(tmp)

        return hessian_mean
