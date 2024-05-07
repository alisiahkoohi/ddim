import torch


def noise_estimation_loss(model, hyper_net,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # Forward pass through the hypernetwork to obtain the
    # parameters of the downstream network.
    parameters_dict = hyper_net(x)

    # Predict the score at this noise level.
    output = torch.func.functional_call(
        model,
        parameters_dict,
        (x, t.float()),
    )

    # output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
