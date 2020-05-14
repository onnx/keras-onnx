###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################


def get_permutation_config(n_dims):
    input_perm_axes = [0, n_dims + 1] + list(range(1, n_dims + 1))
    output_perm_axes = [0] + list(range(2, n_dims + 2)) + [1]
    return input_perm_axes, output_perm_axes
