import torch
from stochman import nnj


def convert_to_stochman(sequential):

    model = []
    for layer in sequential:

        if layer.__class__.__name__ == "Linear":
            stochman_layer = getattr(nnj, layer.__class__.__name__)(layer.in_features, layer.out_features)
            stochman_layer.weight.data = layer.weight.data
            stochman_layer.bias.data = layer.bias.data
        elif layer.__class__.__name__ == "Conv2d":
            stochman_layer = getattr(nnj, layer.__class__.__name__)(
                layer.in_channel, layer.out_channel, layer.kernel
            )
            stochman_layer.weight.data = layer.weight.data
            stochman_layer.bias.data = layer.bias.data
        else:
            stochman_layer = getattr(nnj, layer.__class__.__name__)()

        model.append(stochman_layer)

    model = nnj.Sequential(*model, add_hooks=True)

    return model


class __Dist2__(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, p0: torch.Tensor, p1: torch.Tensor):
        with torch.no_grad():
            with torch.enable_grad():
                C, success = M.connecting_geodesic(p0, p1)
            _, _, dist = C.constant_speed(M)  # B
            dist2 = dist**2

            lm0 = C.deriv(torch.zeros(1, device=p0.device)).squeeze(1)  # log(p0, p1); Bx(d)
            lm1 = -C.deriv(torch.ones(1, device=p0.device)).squeeze(1)  # log(p1, p0); Bx(d)
            G0 = M.metric(p0)  # Bx(d)x(d) or Bx(d)
            G1 = M.metric(p1)  # Bx(d)x(d) or Bx(d)
            if G0.ndim == 3:  # metric is square
                Glm0 = G0.bmm(lm0.unsqueeze(-1)).squeeze(-1)  # Bx(d)
                Glm1 = G1.bmm(lm1.unsqueeze(-1)).squeeze(-1)  # Bx(d)
            else:
                Glm0 = G0 * lm0  # Bx(d)
                Glm1 = G1 * lm1  # Bx(d)

            length_from_log = (lm0 * Glm0).sum(dim=-1)  # B
            alpha = (dist2 / length_from_log).sqrt().unsqueeze(1)  # Bx1
            Glm0 *= alpha
            Glm1 *= alpha

            ctx.save_for_backward(Glm0, Glm1)
        return dist2

    @staticmethod
    def backward(ctx, grad_output):
        Glm0, Glm1 = ctx.saved_tensors
        return (None, 2.0 * grad_output.view(-1, 1) * Glm0, 2.0 * grad_output.view(-1, 1) * Glm1)


def squared_manifold_distance(manifold, p0: torch.Tensor, p1: torch.Tensor):
    """
    Computes the squared distance between point p0 and p1 on the manifold

    Args:
        manifold: manifold object
        p0: initial point
        p1: end point
    Returns:
        dist: squared distance from p0 to p1 calculated on the manifold
    """
    distance_op = __Dist2__()
    return distance_op.apply(manifold, p0, p1)


def tensor_reduction(x: torch.Tensor, reduction: str):
    if reduction == "sum":
        return x.sum()
    elif reduction == "mean":
        return x.mean()
    elif reduction is None or reduction == "none":
        return x
    else:
        raise ValueError(
            f"Expected `reduction` to either be `'mean'`, `'sum'`, `'none'` or `None` but got {reduction}"
        )
