# %%
!pip install einops
!pip install numpy

# %%
!pip install torch
# %%

!pip install matplotlib
# %%
!pip install --upgrade pip
# %% 
! pip install plotly
# %%
!pip install nbformat

# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")

# %%
#print(arr.shape)
print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch
# %%
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row
# %%
arr1=einops.rearrange(arr, "b c h w -> c (b h) w")
print(arr1.shape)
display_array_as_img(arr1)  # plotting all images, stacked in a column

# %%
arr2 = einops.repeat(arr[0],"c h w -> c (2 h) w")
display_array_as_img(arr2)
# %%
#arr3 = einops.repeat(einops.rearrange(arr[:2], "b c h w -> c (b h) w"), "c h w -> c h (2 w)")
arr3 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)

# %%
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)
# %%
arr5 = einops.rearrange(arr[0], "c h w ->  h (c w)")
display_array_as_img(arr5)
# %%
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)
# %%
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)

# %%
arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2 =2, w2 = 2, b1 = 2)
display_array_as_img(arr8)
# %%
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")
# %%
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    return einops.rearrange(t.arange(3,9),"(r c) -> r c", r = 3)
    raise NotImplementedError()


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)
# %%
#Exercise A2

def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1,7),"(r c) -> r c", r = 2)
    raise NotImplementedError()


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))
# %%
# Exercise B1

def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(weeks weekly) -> weeks", "mean", weekly=7)
    raise NotImplementedError()


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))
# %%

# Exercise B2
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    return temps - einops.repeat(temperatures_average(temps), "w -> (w 7)")
    raise NotImplementedError()


expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%
#Exercise B3
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """

    avg1 = einops.reduce(temps, "(w 7) -> w", "mean")
    std1 = einops.reduce(temps, "(w 7) -> w", t.std)
    return (temps - einops.repeat(avg1, "w -> (w 7)"))/(einops.repeat(std1, "w -> (w 7)"))
    raise NotImplementedError()


expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))
# %%

def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    raise NotImplementedError()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)

#%%
# Exercise C1

def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    row_norms = matrix.norm(dim = 1, keepdim=True)
    return matrix/row_norms
    raise NotImplementedError()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)
# %%

def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    matrix_normalized = matrix/matrix.norm(dim = 1, keepdim=True)
    return matrix_normalized @ (matrix_normalized.T)
    raise NotImplementedError()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)
# %%
#Exercise D

def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """

    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    return (t.rand(n,1)>t.cumsum(probs, dim = 0)).sum(dim = 1)
    raise NotImplementedError()


n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)

# %%
einops.repeat(t.Tensor([1,2,3]), "k -> k 2")
A = t.Tensor([1,2,3])
t.sum(A>0, dims = 0)
# %%
#Exercise E

def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    return (t.argmax(scores, dim = 1) == true_classes).float().mean()

    raise NotImplementedError()


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")
# %%
# Exercise F1
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    #return t.gather(prices, 0, items).sum(dim = 0)
    return prices[items].sum().item()
    raise NotImplementedError()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0
print("Tests passed!")




# %%
#Exercise F2

