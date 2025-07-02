""" Visualize the probability density of a 1-D Gaussian Mixture Model and its confidence intervals. 
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from skytraffic.models import GMMPredictionHead

sns.set_style('darkgrid')

def gaussian_pdf(mean, var, x):
    return torch.exp(-0.5 * ((x - mean) ** 2) / var) / torch.sqrt(2 * torch.pi * var)

def weighted_pdf(means, variances, weights, xs):
    return weights[:, None] * gaussian_pdf(means[:, None], variances[:, None], xs[None, :])

def mixture_pdf(means, variances, weights, xs):
    return weighted_pdf(means, variances, weights, xs).sum(axis=-2)

# Parameters for the Gaussian components
means = torch.tensor([-3, 0, 2])          # Means of the components
variances = torch.tensor([0.5, 1, 0.9])  # Variances of the components
mixing = torch.tensor([0.3, 0.6, 0.1])   # Weights (must sum to 1)

# Generate points along the x-axis
xmin, xmax, n_points = -5, 5, 1000
xs = torch.linspace(xmin, xmax, n_points)
dx = abs(xmin - xmax) / n_points

component_densities = gaussian_pdf(means[:, None], variances[:, None], xs[None, :])
weighted_densities = mixing[:, None] * component_densities
mixture_density = (weighted_densities).sum(axis=-2)

# plot vertical lines for the confidence intervals
for confidence, color in zip([0.3, 0.5, 0.7, 0.9], ['darkred', 'orange', 'darkgreen', 'indigo']):
    
    # plot mixture density along with the component densities
    plt.figure(figsize=(8, 4))
    plt.plot(xs, mixture_density.numpy(), '-', linewidth=2, label='Mixture Density')
    for i, comp in enumerate(weighted_densities.numpy()):
        plt.plot(xs, comp, '--', label=f'Component {i+1}')

    lb, ub = GMMPredictionHead.get_confidence_interval(mixing.reshape(1,1,1,-1), 
                                                       means.reshape(1,1,1,-1), 
                                                       torch.log(variances.reshape(1,1,1,-1)), 
                                                       xs, confidence)
    
    for vl, vu in zip(lb.squeeze().tolist(), ub.squeeze().tolist()):
        if vl == vu:
            continue
        xs_in_interval = xs[(xs >= vl) & (xs <= vu)]
        plt.fill_between(xs_in_interval, 0, mixture_density[(xs >= vl) & (xs <= vu)], alpha=0.3, color=color)
        plt.axvline(vl, linestyle='-.', color=color, label='{}%_interval'.format(int(confidence*100)))
        plt.axvline(vu, linestyle='-.', color=color, label='{}%_interval'.format(int(confidence*100)))

    plt.title('1D Gaussian Mixture Model Density')
    plt.xlabel('x')
    plt.ylabel('Probability Density')

    # display legends, but remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gmm_density {} interval.pdf'.format(int(confidence*100)))