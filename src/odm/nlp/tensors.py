import torch
import numpy as np
import matplotlib.pyplot as plt


def mean_tensor(tensors):

    sum_tensor = 0*tensors[0]

    for tensor in tensors:
        sum_tensor += tensor

    return sum_tensor/len(tensors)


def normalize_tensors(tensors):

    center = mean_tensor(tensors)

    return [tensor - center for tensor in tensors]


def PCA(tensors, k=2):

    # preprocessing
    tensor = torch.squeeze(torch.stack(tensors))
    mean = torch.mean(tensor, 0)
    tensor = tensor - mean.expand_as(tensor)

    # SVD
    U, S, V = torch.svd(torch.t(tensor), some=False)
    tensor = torch.mm(tensor, U[:, :k])

    # post processing
    tensor = torch.squeeze(tensor)
    pc_tensors = torch.split(tensor, 1, 0)
    pc_tensors = [torch.squeeze(tensor) for tensor in pc_tensors]

    return pc_tensors


head = {'head_width': 0.1, 'head_length': 0.2}
solid = {'linestyle': 'solid'}
dashed = {'linestyle': 'dashed'}
dotted = {'linestyle': 'dotted'}

arrow_styles = {
  '->': {**solid, **head},
  '-': {**solid},
  '-->': {**dashed, **head},
  '--': {**dashed},
  '..': {**dotted},
  '..>': {**dotted, **head}
}


def plot_embeddings(lines, tensors, arrows):

    vectors = [tensor.data.numpy() for tensor in tensors]
    filename = '/data/fig.png'

    fig, ax = plt.subplots()
    factor = 2
    fig.figsize = (factor*6.4, factor*4.8)

    x = [vec[0] for vec in vectors]
    y = [vec[1] for vec in vectors]

    ax.scatter(x, y, color='white')

    gap = 0.5
    for arrow in arrows:
        i, j, style = arrow
        delta = vectors[j] - vectors[i]
        delta_unit = delta/np.linalg.norm(delta)
        base = vectors[i] + gap*delta_unit
        diff = delta - 2*gap*delta_unit
        plt.arrow(base[0], base[1], diff[0], diff[1],
                  color='#3a3a3a',
                  length_includes_head=True, antialiased=True,
                  **arrow_styles[style])

    printed = []
    for i, line in enumerate(lines):
        if line not in printed:
            ax.annotate(line, (x[i], y[i]), ha='center', va='center')
            printed.append(line)

    plt.axis('off')
    plt.savefig(filename)

    return filename
