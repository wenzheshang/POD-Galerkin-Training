# first line: 144
@memory.cache
def cached_build_L(modes_T, V_field, coords, alpha):
    return build_galerkin_matrix(modes_T, V_field, coords, alpha)
