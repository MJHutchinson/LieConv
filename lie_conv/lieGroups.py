import torch
import numpy as np
from lie_conv.utils import export, Named


@export
def norm(x, dim):
    return (x ** 2).sum(dim=dim).sqrt()


class LieGroup(object, metaclass=Named):
    """The abstract Lie Group requiring additional implementation of exp,log, and lifted_elems
    to use as a new group for LieConv. rep_dim,lie_algebra_dim,q_dim should additionally be specified."""

    rep_dim = NotImplemented  # dimension on which G acts. (e.g. 2 for SO(2))
    lie_algebra_dim = (
        NotImplemented  # dimension of the lie algebra of G. (e.g. 1 for SO(2))
    )
    pseudo_dim = NotImplemented  # dimension of the pseudo embedding
    q_dim = NotImplemented  # dimension which the quotient space X/G is embedded. (e.g. 1 for SO(2) acting on R2)

    def __init__(self, alpha=0.2, debug_config=None, use_pseudo=False, nsamples=None):
        super().__init__()
        self.alpha = alpha
        self.use_pseudo = use_pseudo
        self.debug_config = (
            {"ensure_thetas_in_range": True, "add_random_offsets": True, "tol": 7e-3}
            if debug_config is None
            else debug_config
        )  # added this 7e-3 which was the default sitting here before but need 7e-8 for diagnostics in double precision?
        self.nsamples = nsamples

    @property  # TODO: rename something more sensible
    def lie_dim(self):
        if self.use_pseudo:
            return self.pseudo_dim
        else:
            return self.lie_algebra_dim

    def exp(self, a):
        """Computes (matrix) exponential Lie algebra elements (in a given basis).
        ie out = exp(\sum_i a_i A_i) where A_i are the exponential generators of G.
        Input: [a (*,lie_algebra_dim)] where * is arbitrarily shaped
        Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""
        raise NotImplementedError

    def log(self, u):
        """Computes (matrix) logarithm for collection of matrices and converts to Lie algebra basis.
        Input [u (*,rep_dim,rep_dim)]
        Output [coeffs of log(u) in basis (*,d)]"""
        raise NotImplementedError

    def lifted_elems(self, xyz, nsamples):
        """Takes in coordinates xyz and lifts them to Lie algebra elements a (in basis)
        and embedded orbit identifiers q. For groups where lifting is multivalued
        specify nsamples>1 as number of lifts to do for each point.
        Inputs: [xyz (*,n,rep_dim)],[mask (*,n)], [mask (int)]
        Outputs: [a (*,n*nsamples,lie_algebra_dim)],[q (*,n*nsamples,q_dim)]"""
        raise NotImplementedError

    def inv(self, g):
        """ We can compute the inverse of elements g (*,rep_dim,rep_dim) as exp(-log(g))"""
        return self.exp(-self.log(g))

    def distance(self, abq_pairs):
        """Compute distance of size (*) from [abq_pairs (*,lie_algebra_dim+2*q_dim)].
        Simply computes alpha*norm(log(v^{-1}u)) +(1-alpha)*norm(q_a-q_b),
        combined distance from group element distance and orbit distance."""
        ab_dist = norm(abq_pairs[..., : self.lie_algebra_dim], dim=-1)
        qa = abq_pairs[..., self.lie_algebra_dim : self.lie_algebra_dim + self.q_dim]
        qb = abq_pairs[
            ...,
            self.lie_algebra_dim + self.q_dim : self.lie_algebra_dim + 2 * self.q_dim,
        ]
        qa_qb_dist = norm(qa - qb, dim=-1)
        return ab_dist * self.alpha + (1 - self.alpha) * qa_qb_dist

    def lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
        returns (a,v) with shapes [(*,n*nsamples,lie_algebra_dim),(*,n*nsamples,c)"""
        if self.use_pseudo:
            return self.pseudo_lift(x, nsamples, **kwargs)

        p, v, m = x
        expanded_a, expanded_q = self.lifted_elems(
            p, nsamples, **kwargs
        )  # (bs,n*ns,d), (bs,n*ns,qd)
        nsamples = expanded_a.shape[-2] // m.shape[-1]
        # expand v and mask like q
        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1)
        )  # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(
            *expanded_a.shape[:-1], v.shape[-1]
        )  # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[..., None].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples,)
        )  # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(
            *expanded_a.shape[:-1]
        )  # (bs,n,ns) -> (bs,n*ns)
        # convert from elems to pairs
        paired_a = self.elems2pairs(expanded_a)  # (bs,n*ns,d) -> (bs,n*ns,n*ns,d)
        if expanded_q is not None:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1], 1)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1], 1)
            embedded_locations = torch.cat([paired_a, q_in, q_out], dim=-1)
        else:
            embedded_locations = paired_a
        return (embedded_locations, expanded_v, expanded_mask)

    def pseudo_exp(self, a):
        """Computes takes a parametrisation of the group and returns the matrix representing
        that group element.
        Input: [a (*,lie_algebra_dim)] where * is arbitrarily shaped
        Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""
        raise NotImplementedError

    def pseudo_log(self, u):
        """Compute some parametrisation of the group from matrix elements of the group. To be
        a numerically nicer parametrisation that the lie algebra
        Input [u (*,rep_dim,rep_dim)]
        Output [coeffs of log(u) in basis (*,d)]"""
        raise NotImplementedError

    def pseudo_lifted_elems(self, x, nsamples):
        """Takes in coordinates xyz and lifts them to Lie group elements
        and embedded orbit identifiers q. For groups where lifting is multivalued
        specify nsamples>1 as number of lifts to do for each point.
        Inputs: [xyz (*,n,rep_dim)],[mask (*,n)], [mask (int)]
        Outputs: [a (*,n*nsamples,matrix_dim,matrix_dim)],[q (*,n*nsamples,q_dim)]"""
        raise NotImplementedError

    def pseudo_inv(self, x):
        raise NotImplementedError

    def pseudo_mul(self, x_1, x_2):
        raise NotImplementedError

    def pseudo_lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
        returns (a,v) with shapes [(*,n*nsamples,lie_algebra_dim),(*,n*nsamples,c)"""
        p, v, m = x
        expanded_a, expanded_q = self.pseudo_lifted_elems(
            p, nsamples, **kwargs
        )  # (bs,n*ns,d), (bs,n*ns,qd)
        nsamples = expanded_a.shape[-2] // m.shape[-1]
        # expand v and mask like q
        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1)
        )  # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(
            *expanded_a.shape[:-1], v.shape[-1]
        )  # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[..., None].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples,)
        )  # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(
            *expanded_a.shape[:-1]
        )  # (bs,n,ns) -> (bs,n*ns)
        # convert from elems to pairs by explicitly computing the matrix inverse
        vinv = self.pseudo_inv(expanded_a).unsqueeze(-3)
        u = expanded_a.unsqueeze(-2)
        paired_a = self.pseudo_mul(vinv, u)  # (bs,n*ns,n*ns,d)
        if expanded_q is not None:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1], 1)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1], 1)
            embedded_locations = torch.cat([paired_a, q_in, q_out], dim=-1)
        else:
            embedded_locations = paired_a
        return (embedded_locations, expanded_v, expanded_mask)

    def expand_like(self, v, m, a):
        nsamples = a.shape[-2] // m.shape[-1]
        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1)
        )  # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(
            *a.shape[:2], v.shape[-1]
        )  # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[..., None].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples,)
        )  # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(*a.shape[:2])  # (bs,n,ns) -> (bs,n*ns)
        return expanded_v, expanded_mask

    def elems2pairs(self, a):
        """computes log(e^-b e^a) for all a b pairs along n dimension of input.
        inputs: [a (bs,n,d)] outputs: [pairs_ab (bs,n,n,d)]"""
        vinv = self.exp(-a.unsqueeze(-3))
        u = self.exp(a.unsqueeze(-2))
        return self.log(
            vinv @ u
        )  # ((bs,1,n,d) -> (bs,1,n,r,r))@((bs,n,1,d) -> (bs,n,1,r,r))

    def BCH(self, a, b, order=2):
        """ Baker Campbell Hausdorff formula"""
        assert order <= 4, "BCH only supported up to order 4"
        B = self.bracket
        z = a + b
        if order == 1:
            return z
        ab = B(a, b)
        z += (1 / 2) * ab
        if order == 2:
            return z
        aab = B(a, ab)
        bba = B(b, -ab)
        z += (1 / 12) * (aab + bba)
        if order == 3:
            return z
        baab = B(b, aab)
        z += -(1 / 24) * baab
        return z

    def bracket(self, a, b):
        """Computes the lie bracket between a and b, assumes a,b expressed as vectors"""
        A = self.components2matrix(a)
        B = self.components2matrix(b)
        return self.matrix2components(A @ B - B @ A)

    def __str__(self):
        return (
            f"{self.__class__}({self.alpha})"
            if self.alpha != 0.2
            else f"{self.__class__}"
        )

    def __repr__(self):
        return str(self)


@export
def LieSubGroup(liegroup, generators):
    class subgroup(liegroup):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.orig_dim = self.lie_algebra_dim
            self.lie_algebra_dim = len(generators)
            self.q_dim = self.orig_dim - len(generators)

        def exp(self, a_small):
            a_full = torch.zeros(
                *a_small.shape[:-1],
                self.orig_dim,
                device=a_small.device,
                dtype=a_small.dtype,
            )
            a_full[..., generators] = a_small
            return super().exp(a_full)

        def log(self, U):
            return super().log(U)[..., generators]

        def components2matrix(self, a_small):
            a_full = torch.zeros(
                *a_small.shape[:-1],
                self.orig_dim,
                device=a_small.device,
                dtype=a_small.dtype,
            )
            a_full[..., generators] = a_small
            return super().components2matrix(a_full)

        def matrix2components(self, A):
            return super().matrix2components(A)[..., generators]

        def lifted_elems(self, pt, nsamples=1):
            """pt (bs,n,D) mask (bs,n), per_point specifies whether to
            use a different group element per atom in the molecule"""
            a_full, q = super().lifted_elems(pt, nsamples)
            a_sub = a_full[..., generators]
            complement_generators = list(set(range(self.orig_dim)) - set(generators))
            new_qs = a_full[..., complement_generators]
            q_sub = torch.cat([q, new_qs], dim=-1) if q is not None else new_qs
            return a_sub, q_sub

        # def __str__(self):
        #     return f"Subgroup({str(liegroup)},{generators})"

    return subgroup


@export
class T(LieGroup):
    def __init__(self, k):
        """ Returns the k dimensional translation group. Assumes lifting from R^k"""
        super().__init__()
        self.q_dim = 0
        self.rep_dim = k  # dimension on which G acts
        self.lie_algebra_dim = k  # dimension that g is embedded into

    def lifted_elems(self, xyz, nsamples, **kwargs):
        assert nsamples == 1, "Abelian group, no need for nsamples"
        return xyz, None

    def elems2pairs(self, a):
        deltas = a.unsqueeze(-2) - a.unsqueeze(-3)
        return deltas

    # def distance(self,embedded_pairs):
    #     return norm(embedded_pairs,dim=-1)


# Helper functions for analytic exponential maps. Uses taylor expansions near x=0
# See http://ethaneade.com/lie_groups.pdf for derivations.
thresh = 7e-2


def sinc(x):
    """ sin(x)/x """
    x2 = x * x
    usetaylor = x.abs() < thresh
    return torch.where(
        usetaylor, 1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42)), x.sin() / x
    )


def sincc(x):
    """ (1-sinc(x))/x^2"""
    x2 = x * x
    usetaylor = x.abs() < thresh
    return torch.where(
        usetaylor,
        1 / 6 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72))),
        (x - x.sin()) / x ** 3,
    )


def cosc(x):
    """ (1-cos(x))/x^2"""
    x2 = x * x
    usetaylor = x.abs() < thresh
    return torch.where(
        usetaylor,
        1 / 2 * (1 - x2 / 12 * (1 - x2 / 30 * (1 - x2 / 56))),
        (1 - x.cos()) / x ** 2,
    )


def coscc(x):
    """  """
    x2 = x * x
    # assert not torch.any(torch.isinf(x2)), f"infs in x2 log"
    usetaylor = x.abs() < thresh
    texpand = 1 / 12 * (1 + x2 / 60 * (1 + x2 / 42 * (1 + x2 / 40)))
    costerm = (2 * (1 - x.cos())).clamp(min=1e-6)
    full = (1 - x * x.sin() / costerm) / x ** 2  # Nans can come up here when cos = 1
    output = torch.where(usetaylor, texpand, full)
    return output


def sinc_inv(x):
    usetaylor = x.abs() < thresh
    texpand = 1 + (1 / 6) * x ** 2 + (7 / 360) * x ** 4
    assert not torch.any(
        torch.isinf(texpand) | torch.isnan(texpand)
    ), "sincinv texpand inf" + torch.any(torch.isinf(texpand))
    return torch.where(usetaylor, texpand, x / x.sin())


## Lie Groups acting on R2


@export
class SO2(LieGroup):
    lie_algebra_dim = 1
    pseudo_dim = 2
    rep_dim = 2
    q_dim = 1

    def exp(self, a):
        R = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        sin = a[..., 0].sin()
        cos = a[..., 0].cos()
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        return R

    def log(self, R):
        theta = torch.atan2(R[..., 1, 0] - R[..., 0, 1], R[..., 0, 0] + R[..., 1, 1])[
            ..., None
        ]

        if self.debug_config["ensure_thetas_in_range"]:
            tol = self.debug_config["tol"]
            theta = torch.where(
                torch.abs(theta + np.pi) < tol, theta + 2 * np.pi, theta
            )

            assert (
                (-np.pi + tol < theta)
            ).all(), f"Thetas are not in (-pi, pi]. Error at lower bound: Min is {theta.min()}. Max is {theta.max()}."
            assert (
                (theta <= np.pi + tol)
            ).all(), f"Thetas are not in (-pi, pi]. Error at upper bound: Min is {theta.min()}. Max is {theta.max()}."

        return theta

    def components2matrix(self, a):  # a: (*,lie_dim)
        A = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        A[..., 0, 1] = -a[..., 0]
        A[..., 1, 0] = a[..., 0]
        return A

    def matrix2components(self, A):  # A: (*,rep_dim,rep_dim)
        a = torch.zeros(*A.shape[:-1], 1, device=A.device, dtype=A.dtype)
        a[..., :1] = (A[..., 1, :1] - A[..., :1, 1]) / 2
        return a

    def lifted_elems(self, pt, nsamples=1):
        """pt (bs,n,D) mask (bs,n), per_point specifies whether to
        use a different group element per atom in the molecule"""
        assert nsamples == 1, "Abelian group, no need for nsamples"
        bs, n, D = pt.shape[:3]  # origin = [1,0]
        assert D == 2, "Lifting from R^2 to SO(2) supported only"
        r = norm(pt, dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[..., 1], pt[..., 0]).unsqueeze(-1)
        return theta, r  # checked that lifted_elem(v)@[0,1] = v

    def distance(self, abq_pairs):
        angle_pairs = abq_pairs[..., 0]
        ra = abq_pairs[..., 1]
        rb = abq_pairs[..., 2]
        return angle_pairs.abs() * self.alpha + (1 - self.alpha) * (ra - rb).abs() / (
            ra + rb + 1e-3
        )

    def pseudo_log(self, g):
        """convert 2x2 rotation matrices into embedded vectors of [cos(theta), sin(theta)]"""
        return g[..., 0]

    def pseudo_exp(self, a):
        """convert [cos(theta), sin(theta)] embedding into 2x2 rotation matrix"""
        R = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        cos = a[..., 0]
        sin = a[..., 1]
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        return R

    def pseudo_inv(self, g):
        """Invert the [cos(theta), sin(theta)] embedding"""
        g = g.clone().detach()
        g[..., 1] = -g[..., 1]
        return g

    def pseudo_mul(self, a_1, a_2):
        """Multiply two sets of elements represented as [cos(theta), sin(theta)]"""
        return (self.pseudo_exp(a_1) @ a_2.unsqueeze(-1)).squeeze(-1)

    def pseudo_lifted_elems(self, pt, nsamples):
        """pt (bs,n,D) mask (bs,n), per_point specifies whether to
        use a different group element per atom in the molecule"""
        assert nsamples == 1, "Abelian group, no need for nsamples"
        bs, n, D = pt.shape[:3]  # origin = [1,0]
        assert D == 2, "Lifting from R^2 to SO(2) supported only"
        r = norm(pt, dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[..., 1], pt[..., 0])
        return (
            torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
            r,
        )  # checked that lifted_elem(v)@[0,1] = v


@export
class RxSO2(LieGroup):
    """ Rotation scaling group. Equivalent to log polar convolution."""

    lie_algebra_dim = 2
    rep_dim = 2
    q_dim = 0

    def exp(self, a):
        logr = a[..., 0]
        R = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        rsin = logr.exp() * a[..., 1].sin()
        rcos = logr.exp() * a[..., 1].cos()
        R[..., 0, 0] = rcos
        R[..., 1, 1] = rcos
        R[..., 0, 1] = -rsin
        R[..., 1, 0] = rsin
        return R

    def log(self, R):
        rsin = (R[..., 1, 0] - R[..., 0, 1]) / 2
        rcos = (R[..., 0, 0] + R[..., 1, 1]) / 2
        theta = torch.atan2(rsin, rcos)
        r = (rsin ** 2 + rcos ** 2).sqrt()
        return torch.stack([r.log(), theta], dim=-1)

    def lifted_elems(self, pt, nsamples=1):
        bs, n, D = pt.shape[:3]  # origin = [1,0]
        assert D == 2, "Lifting from R^2 to RxSO(2) supported only"
        r = norm(pt, dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[..., 1], pt[..., 0]).unsqueeze(-1)
        return torch.cat([r.log(), theta], dim=-1), None

    def distance(self, abq_pairs):
        angle_dist = abq_pairs[..., 1].abs()
        r_dist = abq_pairs[..., 0].abs()
        return angle_dist * self.alpha + (1 - self.alpha) * r_dist


@export
class RxSQ(LieGroup):
    """Rotation Squeeze group. Equivalent to log hyperbolic coordinate convolution.
    Acts on the positive orthant R2+."""

    lie_algebra_dim = 2
    rep_dim = 2
    q_dim = 0

    def exp(self, a):
        raise NotImplementedError

    def log(self, R):
        raise NotImplementedError

    def lifted_elems(self, pt, nsamples=1):
        bs, n, D = pt.shape[:3]  # origin = [1,0]
        assert nsamples == 1, "Abelian group, no need for nsamples"
        assert D == 2, "Lifting from R^2 to RxSQ supported only"
        lxy = pt.log()
        logs = (lxy[..., 0] - lxy[..., 1]) / 2
        logr = (lxy[..., 0] + lxy[..., 1]) / 2
        return torch.cat([logr, logs], dim=-1), None

    def distance(self, abq_pairs):
        s_dist = abq_pairs[..., 1].abs()
        r_dist = abq_pairs[..., 0].abs()
        return s_dist * self.alpha + (1 - self.alpha) * r_dist


@export
class Rx(LieSubGroup(RxSO2, (0,))):
    pass


@export
class SQ(LieSubGroup(RxSQ, (1,))):
    pass


@export
class Tx(LieSubGroup(T, (0,))):
    pass


@export
class Ty(LieSubGroup(T, (1,))):
    pass


@export
class SE2(SO2):
    lie_algebra_dim = 3
    pseudo_dim = 4
    rep_dim = 3
    q_dim = 0

    def log(self, g):
        theta = super().log(g[..., :2, :2])
        I = torch.eye(2, device=g.device, dtype=g.dtype)
        K = super().components2matrix(torch.ones_like(theta))
        theta = theta.unsqueeze(-1)
        Vinv = (sinc(theta) / (2 * cosc(theta))) * I - theta * K / 2
        a = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
        a[..., 0] = theta[..., 0, 0]
        a[..., 1:] = (Vinv @ g[..., :2, 2].unsqueeze(-1)).squeeze(-1)
        return a

    def exp(self, a):
        """assumes that a is expanded in the basis [tx,ty,theta] of the lie algebra
        a should have shape (*,3)"""
        theta = a[..., 0].unsqueeze(-1)
        I = torch.eye(2, device=a.device, dtype=a.dtype)
        K = super().components2matrix(torch.ones_like(a))
        theta = theta.unsqueeze(-1)
        V = sinc(theta) * I + theta * cosc(theta) * K
        g = torch.zeros(*a.shape[:-1], 3, 3, device=a.device, dtype=a.dtype)
        g[..., :2, :2] = theta.cos() * I + theta.sin() * K
        g[..., :2, 2] = (V @ a[..., 1:].unsqueeze(-1)).squeeze(-1)
        g[..., 2, 2] = 1
        return g

    def components2matrix(self, a):
        """takes an element in the lie algebra expressed in the standard basis and
        expands to the corresponding matrix. a: (*,3)"""
        A = torch.zeros(*a.shape, 3, device=a.device, dtype=a.dtype)
        A[..., 2, :2] = a[..., 1:]
        A[..., 0, 1] = a[..., 0]
        A[..., 1, 0] = -a[..., 0]
        return A

    def matrix2components(self, A):
        """takes an element in the lie algebra expressed as a matrix (*,3,3) and
        expresses it in the standard basis"""
        a = torch.zeros(*A.shape[:-1], device=A.device, dtype=A.dtype)
        a[..., 1:] = A[..., :2, 2]
        a[..., 0] = (A[..., 1, 0] - A[..., 0, 1]) / 2
        return a

    def lifted_elems(self, pt, nsamples=1):
        # TODO: correctly handle masking, unnecessary for image data
        d = self.rep_dim
        # Sample stabilizer of the origin
        # thetas = (torch.rand(*p.shape[:-1],1).to(p.device)*2-1)*np.pi
        # thetas = torch.randn(nsamples)*2*np.pi - np.pi
        thetas = torch.linspace(
            -np.pi, np.pi, nsamples + 1, device=pt.device, dtype=pt.dtype
        )[
            1:
        ]  # .to(pt.device)
        for _ in pt.shape[:-1]:  # uniform on circle, but -pi and pi ar the same
            thetas = thetas.unsqueeze(0)
        if self.debug_config["add_random_offsets"]:
            thetas = (
                thetas
                + torch.rand(*pt.shape[:-1], 1, device=pt.device, dtype=pt.dtype)
                * 2
                * np.pi
            )  # .to(pt.device)
        R = torch.zeros(
            *pt.shape[:-1], nsamples, d, d, device=pt.device, dtype=pt.dtype
        )  # .to(pt.device)
        sin, cos = thetas.sin(), thetas.cos()
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        R[..., 2, 2] = 1
        # Get T(p)
        T = torch.zeros_like(R)
        T[..., 0, 0] = 1
        T[..., 1, 1] = 1
        T[..., 2, 2] = 1
        T[..., :2, 2] = pt.unsqueeze(-2)
        flat_a = self.log(T @ R).reshape(*pt.shape[:-2], pt.shape[-2] * nsamples, d)
        return flat_a, None

    def distance(self, abq_pairs):
        d_theta = abq_pairs[..., 0].abs()
        d_r = norm(abq_pairs[..., 1:], dim=-1)
        return d_theta * self.alpha + (1 - self.alpha) * d_r

    def pseudo_log(self, g):
        """convert 3x3 rotation + translation matrix into embedded vectors of [tx, ty, cos(theta), sin(theta)]"""
        a_so2 = super().pseudo_log(g[..., :2, :2])
        a_t = g[..., :2, 2]
        return torch.cat([a_t, a_so2], dim=-1)

    def pseudo_exp(self, a):
        """convert [tx, ty, cos(theta), sin(theta)] embedding into 2x2 rotation matrix"""
        R = super().pseudo_exp(a[..., 2:])
        g = torch.zeros((*a.shape[:-1], 3, 3), device=a.device, dtype=a.dtype)
        g[..., :2, :2] = R
        g[..., :2, 2] = a[..., :2]
        g[..., 2, 2] = 1
        return g

    def pseudo_inv(self, g):
        """Invert the [tx, ty, cos(theta), sin(theta)] embedding"""
        g[..., 2:] = super().pseudo_inv(g[..., 2:])
        g[..., :2] = -g[..., :2]
        return g

    def pseudo_mul(self, a_1, a_2):
        """Multiply two sets of elements represented as [cos(theta), sin(theta)]"""
        return torch.cat(
            [
                a_1[..., :2] + a_2[..., :2],
                super().pseudo_mul(a_1[..., 2:], a_2[..., 2:]),
            ]
        )

    def pseudo_lifted_elems(self, pt, nsamples):
        """pt (bs,n,D) mask (bs,n), per_point specifies whether to
        use a different group element per atom in the molecule"""
        thetas = torch.linspace(-np.pi, np.pi, nsamples + 1)[1:].to(pt.device)
        for _ in pt.shape[:-1]:  # uniform on circle, but -pi and pi ar the same
            thetas = thetas.unsqueeze(0)
        thetas = thetas + torch.rand(*pt.shape[:-1], 1).to(pt.device) * 2 * np.pi
        a = torch.zeros(*pt.shape[:-1], nsamples, 4).to(pt.device)
        for i in range(nsamples):
            a[..., i, :2] = pt
        a[..., 2] = thetas.cos()
        a[..., 3] = thetas.sin()
        return a.reshape((a.shape[0], -1, a.shape[-1]))


@export
class SE2_canonical(LieGroup):
    lie_dim = 3
    rep_dim = 3
    q_dim = 0

    def matrixify(self, X, nsamples):
        angles = 2 * np.pi * X[..., [2]] / nsamples
        cosines = torch.cos(angles)
        sines = torch.sin(angles)

        rotations_1 = torch.cat([cosines, -sines], dim=2).unsqueeze(2)
        rotations_2 = torch.cat([sines, cosines], dim=2).unsqueeze(2)
        rotations = torch.cat([rotations_1, rotations_2], dim=2)

        X_lift = torch.cat([rotations, X[..., :2].unsqueeze(3)], dim=3)
        X_lift = torch.cat([X_lift, torch.ones_like(X_lift)[:, :, :1, :]], dim=2)
        X_lift[:, :, [2], :2] = 0.0

        return X_lift, rotations

    def lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
        returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
        p, v, m = x
        rotations = torch.arange(nsamples, dtype=p.dtype, device=p.device).repeat(
            (*p.shape[:2])
        )
        rotations = rotations.unsqueeze(-1)
        p_lift = p[..., None, :].repeat((1,) * len(p.shape[:-1]) + (nsamples, 1))
        p_lift = p_lift.reshape(p.shape[0], p.shape[1] * nsamples, p.shape[2])
        X_lift = torch.cat([p_lift, rotations], dim=-1)
        X_pairs = X_lift[..., None, :, :] - X_lift[..., :, None, :]
        _, rotations_inverse = self.matrixify(-X_lift, nsamples)
        rotations_inverse = rotations_inverse.unsqueeze(2).repeat(
            1, 1, rotations_inverse.shape[1], 1, 1
        )
        X_pairs = torch.cat(
            [
                (rotations_inverse @ (X_pairs[..., :2, None])).squeeze(-1),
                torch.remainder(X_pairs[..., [-1]], nsamples),
            ],
            dim=-1,
        )

        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1)
        )  # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(
            *X_lift.shape[:-1], v.shape[-1]
        )  # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[..., None].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples,)
        )  # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(
            *X_lift.shape[:-1]
        )  # (bs,n,ns) -> (bs,n*ns)
        return (X_pairs, expanded_v, expanded_mask)


## Lie Groups acting on R3

# Hodge star on R3
def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1], 3, 3, device=k.device, dtype=k.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    return K


def uncross_matrix(K):
    """Application of hodge star on R3, mapping Λ^2 R3 -> Λ^1 R3"""
    k = torch.zeros(*K.shape[:-1], device=K.device, dtype=K.dtype)
    k[..., 0] = (K[..., 2, 1] - K[..., 1, 2]) / 2
    k[..., 1] = (K[..., 0, 2] - K[..., 2, 0]) / 2
    k[..., 2] = (K[..., 1, 0] - K[..., 0, 1]) / 2
    return k


def quaternion_multiply(a, b):
    # todo: replace with efficient method https://math.stackexchange.com/questions/1103399/alternative-quaternion-multiplication-method
    R = torch.zeros((*a.shape[:-1], 4, 4), dtype=a.dtype, device=a.device)
    R[..., 0] = a

    R[..., 0, 1] = -a[..., 1]
    R[..., 1, 1] = a[..., 0]
    R[..., 2, 1] = a[..., 3]
    R[..., 3, 1] = -a[..., 2]

    R[..., 0, 2] = -a[..., 2]
    R[..., 1, 2] = -a[..., 3]
    R[..., 2, 2] = a[..., 0]
    R[..., 3, 2] = a[..., 1]

    R[..., 0, 3] = -a[..., 3]
    R[..., 1, 3] = a[..., 2]
    R[..., 2, 3] = -a[..., 1]
    R[..., 3, 3] = a[..., 0]

    return (R @ b.unsqueeze(-1)).squeeze()


def quaternion_conjugate(a):
    a = a.clone().detach()
    a[..., 1:] = -a[..., 1:]
    return a


def dual_quaternion_multiply(a, b):
    a1 = a[..., :4]
    a2 = a[..., 4:]
    b1 = b[..., :4]
    b2 = b[..., 4:]

    c1 = quaternion_multiply(a1, b1)
    c2 = quaternion_multiply(a1, b2) + quaternion_multiply(a2, b1)

    return torch.cat([c1, c2], dim=-1)


def dual_conjugate(a):
    a = a.clone().detach()
    a[..., 4:] = -a[..., 4:]
    return a


def dual_quaternion_conjugate(a):
    a = a.clone().detach()
    a[..., 1:4] = -a[..., 1:4]
    a[..., 5:8] = -a[..., 5:8]

    return a


def dual_double_conjugate(a):
    a = a.clone().detach()
    a[..., 1:4] = -a[..., 1:4]
    a[..., 4] = -a[..., 4]

    return a


@export
class SO3(LieGroup):
    lie_algebra_dim = 3
    pseudo_dim = 4
    rep_dim = 3
    q_dim = 1

    def __init__(self, positive_quaternions=False, **kwargs):
        super().__init__(**kwargs)
        self.positive_quaternions = positive_quaternions

    def exp(self, w):
        """Rodriguez's formula, assuming shape (*,3)
        where components 1,2,3 are the generators for xrot,yrot,zrot"""
        theta = norm(w, dim=-1)[..., None, None]
        K = cross_matrix(w)
        I = torch.eye(3, device=K.device, dtype=K.dtype)
        Rs = I + K * sinc(theta) + (K @ K) * cosc(theta)
        return Rs

    def log(self, R):
        """ Computes components in terms of generators rx,ry,rz. Shape (*,3,3)"""
        trR = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        costheta = ((trR - 1) / 2).clamp(max=1, min=-1).unsqueeze(-1)
        theta = torch.acos(costheta)
        logR = uncross_matrix(R) * sinc_inv(theta)
        return logR

    def components2matrix(self, a):  # a: (*,3)
        return cross_matrix(a)

    def matrix2components(self, A):  # A: (*,rep_dim,rep_dim)
        return uncross_matrix(A)

    def sample(self, *shape, device=torch.device("cuda"), dtype=torch.float32):
        q = torch.randn(*shape, 4, device=device, dtype=dtype)
        q /= norm(q, dim=-1).unsqueeze(-1)
        theta_2 = torch.atan2(norm(q[..., 1:], dim=-1), q[..., 0]).unsqueeze(-1)
        so3_elem = (
            2 * sinc_inv(theta_2) * q[..., 1:]
        )  # # (sin(x/2)u -> xu) for x angle and u direction
        R = self.exp(so3_elem)
        return R

    def lifted_elems(self, pt, nsamples, **kwargs):
        """Lifting from R^3 -> SO(3) , R^3/SO(3). pt shape (*,3)
        First get a random rotation Rz about [1,0,0] by the appropriate angle
        and then rotate from [1,0,0] to p/\|p\| with Rp  to get RpRz and then
        convert to logarithmic coordinates log(RpRz), \|p\|"""
        d = self.rep_dim
        device, dtype = pt.device, pt.dtype
        # Sample stabilizer of the origin
        q = torch.randn(*pt.shape[:-1], nsamples, 4, device=device, dtype=dtype)
        q /= norm(q, dim=-1).unsqueeze(-1)
        theta = 2 * torch.atan2(norm(q[..., 1:], dim=-1), q[..., 0]).unsqueeze(-1)
        zhat = torch.zeros(
            *pt.shape[:-1], nsamples, 3, device=device, dtype=dtype
        )  # (*,3)
        zhat[..., 0] = 1  # theta
        Rz = self.exp(zhat * theta)

        # Compute the rotation between zhat and p
        r = norm(pt, dim=-1).unsqueeze(-1)  # (*,1)
        assert not torch.any(torch.isinf(pt) | torch.isnan(pt))
        p_on_sphere = pt / r.clamp(min=1e-5)
        w = torch.cross(zhat, p_on_sphere[..., None, :].expand(*zhat.shape))
        sin = norm(w, dim=-1)
        cos = p_on_sphere[..., None, 0]

        angle = torch.atan2(sin, cos).unsqueeze(-1)  # cos angle
        Rp = self.exp(w * sinc_inv(angle))

        # Combine the rotations into one
        A = self.log(Rp @ Rz)  # Convert to lie algebra element
        assert not torch.any(torch.isnan(A) | torch.isinf(A))
        q = r[..., None, :].expand(
            *r.shape[:-1], nsamples, 1
        )  # The orbit identifier is \|x\|
        flat_q = q.reshape(*r.shape[:-2], r.shape[-2] * nsamples, 1)
        flat_a = A.reshape(*pt.shape[:-2], pt.shape[-2] * nsamples, d)
        return flat_a, flat_q

    def pseudo_log(self, g):
        """convert 3x3 rotation matrices into quaternion representation"""
        trR = g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
        costheta = ((trR - 1) / 2).clamp(max=1, min=-1)
        theta = torch.acos(costheta)
        cos2theta = torch.cos(theta / 2)
        sin2theta = torch.sin(theta / 2)
        w = uncross_matrix(g) * sinc_inv(theta.unsqueeze(-1))
        w = w / w.norm(dim=-1).unsqueeze(-1)
        a = torch.zeros((*g.shape[:-2], 4), device=g.device, dtype=g.dtype)
        a[..., 0] = cos2theta
        a[..., 1] = sin2theta * w[..., 0]
        a[..., 2] = sin2theta * w[..., 1]
        a[..., 3] = sin2theta * w[..., 2]
        return a

    def pseudo_exp(self, a):
        """convert quaternion representation embedding into 3x3 rotation matrix"""
        theta = 2 * torch.acos(a[..., 0]).unsqueeze(-1)
        w = a[..., 1:] * theta / torch.sin(theta / 2)
        K = cross_matrix(w)
        I = torch.eye(3, device=K.device, dtype=K.dtype)
        theta = theta.unsqueeze(-1)
        Rs = I + K * sinc(theta) + (K @ K) * cosc(theta)
        return Rs

    def pseudo_inv(self, g):
        """Invert the quaternion representation"""
        g = g.clone().detach()
        g[..., 1:] = -g[..., 1:]
        return g

    def pseudo_mul(self, a_1, a_2):
        """Multiply two sets of elements represented as [cos(theta), sin(theta)]"""
        return quaternion_multiply(a_1, a_2)

    def pseudo_lifted_elems(self, pt, nsamples):
        # Sample the origin stabaliser
        b, n, d = pt.shape
        thetas2 = torch.randn(b, n, nsamples) * np.pi
        # create quaternion rotations about the x axis - stabaliser of [1,0,0]
        qH = torch.zeros(b, n, nsamples, 4, device=pt.device, dtype=pt.dtype)
        qH[..., 0] = thetas2.cos()
        qH[..., 1] = thetas2.sin()  # origin is x xaxis

        pt_on_sphere = pt / pt.norm(dim=-1).unsqueeze(-1).clamp(min=1e-5)
        x_hat = torch.zeros(b, n, nsamples, 3, device=pt.device, dtype=pt.dtype)
        x_hat[..., 0] = 1.0
        # cross procut between origin and point gives axis of rotation
        w = torch.cross(x_hat, pt_on_sphere[..., None, :].repeat(1, 1, nsamples, 1))
        sin = norm(w, dim=-1)
        cos = pt_on_sphere[..., None, 0]

        angle = torch.atan2(sin, cos).unsqueeze(-1)
        qP = torch.zeros(b, n, nsamples, 4, device=pt.device, dtype=pt.dtype)
        qP[..., 0] = (angle / 2).cos().squeeze(-1)
        qP[..., 1:] = (angle / 2).sin() * (w / w.norm(dim=-1).unsqueeze(-1))

        # Multiply coset rep by stabiliser samples
        q = quaternion_multiply(qP, qH)
        if self.positive_quaternions:
            sign = (q[..., 0] > 0).float() * 2 - 1
            q = q * sign.unsqueeze(-1)
        r = pt.norm(dim=-1)[..., None, None].repeat(1, 1, nsamples, 1)

        return q.reshape(b, n * nsamples, 4), r.reshape(b, n * nsamples, 1)


@export
class SE3(SO3):
    lie_algebra_dim = 6
    rep_dim = 4
    q_dim = 0

    def __init__(
        self,
        per_point=True,
        dual_quaternions=True,
        positive_quaternions=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.per_point = per_point
        self.dual_quaternions = dual_quaternions
        self.positive_quaternions = positive_quaternions

    @property
    def pseudo_dim(self):
        if self.dual_quaternions:
            return 8
        else:
            return 7

    def exp(self, w):
        theta = norm(w[..., :3], dim=-1)[..., None, None]
        K = cross_matrix(w[..., :3])
        R = super().exp(w[..., :3])
        I = torch.eye(3, device=w.device, dtype=w.dtype)
        V = I + cosc(theta) * K + sincc(theta) * (K @ K)
        U = torch.zeros(*w.shape[:-1], 4, 4, device=w.device, dtype=w.dtype)
        U[..., :3, :3] = R
        U[..., :3, 3] = (V @ w[..., 3:].unsqueeze(-1)).squeeze(-1)
        U[..., 3, 3] = 1
        return U

    def log(self, U):
        w = super().log(U[..., :3, :3])
        I = torch.eye(3, device=w.device, dtype=w.dtype)
        K = cross_matrix(w[..., :3])
        theta = norm(w, dim=-1)[..., None, None]  #%(2*np.pi)
        # theta[theta>np.pi] -= 2*np.pi
        cosccc = coscc(theta)
        Vinv = I - K / 2 + cosccc * (K @ K)
        u = (Vinv @ U[..., :3, 3].unsqueeze(-1)).squeeze(-1)
        # assert not torch.any(torch.isnan(u)), f"nans in u log {torch.isnan(u).sum()}, {torch.where(torch.isnan(u))}"
        return torch.cat([w, u], dim=-1)

    def components2matrix(self, a):  # a: (*,3)
        A = torch.zeros(*a.shape[:-1], 4, 4, device=a.device, dtype=a.dtype)
        A[..., :3, :3] = cross_matrix(a[..., :3])
        A[..., :3, 3] = a[..., 3:]
        return A

    def matrix2components(self, A):  # A: (*,4,4)
        return torch.cat([uncross_matrix(A[..., :3, :3]), A[..., :3, 3]], dim=-1)

    def lifted_elems(self, pt, nsamples):
        """pt (bs,n,D) mask (bs,n), per_point specifies whether to
        use a different group element per atom in the molecule"""
        # return farthest_lift(self,pt,mask,nsamples,alpha)
        # same lifts for each point right now
        bs, n = pt.shape[:2]
        if self.per_point:
            q = torch.randn(bs, n, nsamples, 4, device=pt.device, dtype=pt.dtype)
        else:
            q = torch.randn(bs, 1, nsamples, 4, device=pt.device, dtype=pt.dtype)
        q /= norm(q, dim=-1).unsqueeze(-1)
        theta_2 = torch.atan2(norm(q[..., 1:], dim=-1), q[..., 0]).unsqueeze(-1)
        so3_elem = (
            2 * sinc_inv(theta_2) * q[..., 1:]
        )  # (sin(x/2)u -> xu) for x angle and u direction
        se3_elem = torch.cat([so3_elem, torch.zeros_like(so3_elem)], dim=-1)
        R = self.exp(se3_elem)
        T = torch.zeros(
            bs, n, nsamples, 4, 4, device=pt.device, dtype=pt.dtype
        )  # (bs,n,nsamples,4,4)
        T[..., :, :] = torch.eye(4, device=pt.device, dtype=pt.dtype)
        T[..., :3, 3] = pt[:, :, None, :]  # (bs,n,1,3)
        a = self.log(T @ R)  # @R) # bs, n, nsamples, 6
        return a.reshape(bs, n * nsamples, 6), None

    def distance(self, abq_pairs):
        dist_rot = norm(abq_pairs[..., :3], dim=-1)
        dist_trans = norm(abq_pairs[..., 3:], dim=-1)
        return dist_rot * self.alpha + (1 - self.alpha) * dist_trans

    def qRt_to_qRqD(self, qRt):
        qR = qRt[..., :4]
        qT = torch.zeros(qR.shape, device=qR.device, dtype=qR.dtype)
        qT[..., 1:] = qRt[..., 4:]
        qD = quaternion_multiply(qT, qR) / 2
        return torch.cat([qR, qD], dim=-1)

    def qRqD_to_qRt(self, qRqD):
        qR = qRqD[..., :4]
        qD = qRqD[..., 4:]
        t = 2 * quaternion_multiply(qD, quaternion_conjugate(qR))[..., 1:]
        return torch.cat([qR, t], dim=-1)

    def pseudo_log(self, g):
        """convert 4x4 se3 matrices into quaternion representation"""
        qR = super().pseudo_log(g[..., :3, :3])
        t = g[..., :3]
        qT = torch.zeros_like(qR)
        qT[..., 1:] = t / 2
        qD = quaternion_multiply(qT, qR)

        l = torch.cat([qR, qD], dim=-1)
        return l

    def pseudo_exp(self, a):
        """convert quaternion representation embedding into 3x3 rotation matrix"""
        qR = a[..., :4]
        qD = a[..., 4:]
        R = super().pseudo_exp(qR)
        t = 2 * quaternion_multiply(qD, self.pseudo_inv(qR))[..., 1:]
        g = torch.zeros((*a.shape[:-1], 4, 4), device=a.device, dtype=a.dtype)
        g[..., :3, :3] = R
        g[..., 3, :3] = t
        g[..., 3, 3] = 1.0
        return g

    def pseudo_inv(self, g):
        """Invert the quaternion representation"""
        g = g.clone().detach()
        g[..., 1:4] = -g[..., 1:4]
        g[..., 5:8] = -g[..., 5:8]
        # g[..., 4] = -g[..., 4]
        return g

    def pseudo_mul(self, a_1, a_2):
        """Multiply two sets of elements represented as [cos(theta), sin(theta)]"""
        return dual_quaternion_multiply(a_1, a_2)

    def pseudo_lifted_elems(self, pt, nsamples):
        bs, n = pt.shape[:2]
        if self.per_point:
            qR = torch.randn(bs, n, nsamples, 4, device=pt.device, dtype=pt.dtype)
        else:
            qR = torch.randn(bs, 1, nsamples, 4, device=pt.device, dtype=pt.dtype)
        qR /= norm(qR, dim=-1).unsqueeze(-1)
        if self.positive_quaternions:
            sign = (qR[..., 0] > 0).float() * 2 - 1
            qR = qR * sign.unsqueeze(-1)
        qT = torch.zeros((*pt.shape[:-1], 4), dtype=pt.dtype, device=pt.device)
        qT[..., 1:] = pt
        qT = qT[..., None, :].repeat_interleave(nsamples, 2)
        qD = quaternion_multiply(qT, qR) / 2
        return torch.cat([qR, qD], dim=-1).reshape(bs, -1, 8), None

    def pseudo_lift(self, x, nsamples, **kwargs):
        g, v, m = super().pseudo_lift(x, nsamples, **kwargs)
        if not self.dual_quaternions:
            g = self.qRqD_to_qRt(g)
        return g, v, m


@export
class Trivial(LieGroup):
    lie_algebra_dim = 0

    def __init__(self, dim=2):
        super().__init__()
        self.q_dim = dim
        self.rep_dim = dim

    def lift(self, x, nsamples, **kwargs):
        assert nsamples == 1, "Abelian group, no need for nsamples"
        p, v, m = x
        bs, n, d = p.shape
        qa = p[..., :, None, :].expand(bs, n, n, d)
        qb = p[..., None, :, :].expand(bs, n, n, d)
        q = torch.cat([qa, qb], dim=-1)
        return q, v, m

    # def distance(self,abq_pairs):
    #     qa = abq_pairs[...,:self.q_dim]
    #     qb = abq_pairs[...,self.q_dim:]
    #     return norm(qa-qb,dim=-1)


@export
class FakeSchGroup(object, metaclass=Named):
    lie_algebra_dim = 0
    rep_dim = 3
    q_dim = 1

    def lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
        returns (a,v) with shapes [(*,n*nsamples,lie_algebra_dim),(*,n*nsamples,c)"""
        p, v, m = x
        q = (p[..., :, None, :] - p[..., None, :, :]).norm(dim=-1).unsqueeze(-1)
        return (q, v, m)

    def distance(self, abq_pairs):
        return abq_pairs
