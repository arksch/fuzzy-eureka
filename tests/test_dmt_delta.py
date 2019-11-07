"""
Testing the delta vs 2delta behavior
"""
import pytest


parametrize = pytest.mark.parametrize


@parametrize("algorithm, deltatrick",
             [
                 ("binning", True),
                 ("binning", False),
                 ("induced", True),
                 ("induced", False),
                 ("gradient", True),
                 ("gradient", False),
             ])
def test_morse_reduce_approx_delta(random_alpha_complex_2D50_samples, algorithm, deltatrick):
    from dmt.pers_hom import bottleneck
    from dmt.dmt import morse_approx_binning, morse_approx_induced_matching, morse_approx_along_gradients
    delta = .05
    algo_map = {"binning": morse_approx_binning,
                "induced": morse_approx_induced_matching,
                "gradient": morse_approx_along_gradients}
    cplx = random_alpha_complex_2D50_samples
    reduced_cplx = algo_map[algorithm](cplx, delta, deltatrick=deltatrick)
    assert reduced_cplx.valid_boundary()
    true_dgm = cplx.persistence_diagram()
    approx_dgm = reduced_cplx.persistence_diagram()
    distance = bottleneck(true_dgm, approx_dgm)
    print(distance)
    assert delta < bottleneck(true_dgm, {}), "Ill posed approximation test if delta is larger " \
                                             "than the distance to the empty diagram"
    assert bottleneck(true_dgm, true_dgm) < 1e-9
    assert bottleneck(approx_dgm, approx_dgm) < 1e-9
    assert 1e-9 < distance <= delta + 1e-9  # Some tolerance


def test_bad_points():
    import numpy as np
    from dmt import morse_approx_induced_matching
    from dmt.complexes import AlphaComplex
    from dmt.pers_hom import bottleneck
    points = np.array([[ 0.47665661,  0.75840924],
               [-0.26799701, -0.579576  ],
               [ 0.50223673, -0.12207289],
               [ 1.89449196,  0.19588077],
               [-1.32996325, -1.2092812 ],
               [-3.64248974, -1.49540268],
               [ 0.7381247 , -1.16459865],
               [ 0.07172006,  1.01562597],
               [-0.96203438,  1.13346885],
               [-1.40299403,  2.54347974],
               [-0.73021045, -0.1229212 ],
               [-0.61439488,  0.49431323],
               [ 0.32618852,  1.17864799],
               [ 1.8283409 , -0.92010186],
               [-1.19754034,  0.85258149],
               [ 0.24564675,  0.98083611],
               [-0.50321869,  0.00942479],
               [ 0.40034828,  0.61182454],
               [-0.61416878,  0.7822097 ],
               [ 0.7964955 ,  0.17743997],
               [ 2.06589058,  0.61075921],
               [-0.97221092, -0.95713679],
               [ 0.00823418,  1.0559076 ],
               [-1.60575126, -0.69568841],
               [ 1.39223893, -0.90224819],
               [ 0.93976872, -0.82296745],
               [ 2.13377113,  1.13844349],
               [-0.81443489, -0.2765248 ],
               [-1.67322648, -0.30944305],
               [ 0.66698462, -0.00367728],
               [ 0.38328809,  0.54237665],
               [ 0.31065489,  1.1439971 ],
               [-0.51447297, -0.19639922],
               [-1.02626235, -0.82863989],
               [ 0.37813997, -0.4588291 ],
               [-0.31763019, -0.73191275],
               [ 0.61494494, -0.91558286],
               [ 0.01534708,  0.86336835],
               [ 0.75551287,  0.11027708],
               [-0.1268198 , -1.11568991],
               [-1.04879115, -1.21261188],
               [ 1.76837027,  0.19542733],
               [ 0.45896721, -0.4811429 ],
               [-0.95285576,  0.94209284],
               [-0.72488572,  0.35963128],
               [ 0.67284079, -0.37688575],
               [-1.28076795,  1.33674396],
               [-0.36398862, -0.32323984],
               [-0.43343937, -0.47182564],
               [-0.50305469,  0.92829217]])
    delta = 0.05
    cplx = AlphaComplex(points)
    reduced_cplx, matching = morse_approx_induced_matching(cplx, delta, return_matching=True)
    distance = bottleneck(cplx.persistence_diagram(), reduced_cplx.persistence_diagram())
    assert 1e-9 < distance < delta + 1e-9
