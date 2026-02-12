from builtins import breakpoint
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from copy import deepcopy


class Identity(nn.Module):
    """Identity module that will return the same input as it receives."""

    def __init__(self):
        super().__init__()

    # def forward(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    #     val = x

    #     if jacobian:
    #         xs = x.shape
    #         jac = (
    #             torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
    #             .repeat(xs[0], 1, 1)
    #             .reshape(xs[0], *xs[1:], *xs[1:])
    #         )
    #         return val, jac
    #     return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            xs = x.shape
            jacobian = (
                torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
                .repeat(xs[0], 1, 1)
                .reshape(xs[0], *xs[1:], *xs[1:])
            )
            return jacobian

    # def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
    #     return jac_in

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return vector

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return vector

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return matrix

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        if wrt == "input":
            return matrix


def identity(x: Tensor) -> Tensor:
    """Function that for a given input x returns the corresponding identity jacobian matrix"""
    m = Identity()
    # return m(x, jacobian=True)[1]
    return m._jacobian(x)


class AbstractJacobian:
    """Abstract class that:
    - will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    - propagate jacobian vector and jacobian matrix products, both forward and backward
    - pull back and push forward metrics
    """

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # return self._jacobian_wrt_input_mult_left_vec(x, val, identity(x))
        # return self._jmp(x, val, identity(x))
        # return self._mjp(x, val, identity(val), wrt = wrt)
        raise NotImplementedError

    ### forward passes ###

    def jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """jacobian vector product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        assert vector.shape == xs
        vector = vector.reshape(xs[0], xs[1:].numel())
        jacobian_vector_product = self._jvp(x, val, vector, wrt=wrt)
        return jacobian_vector_product.reshape(vs[0], *vs[1:])

    def jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """jacobian matrix product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(x)
        else:
            assert matrix.shape == (xs[0], *xs[1:], *xs[1:])
        matrix = matrix.reshape(xs[0], xs[1:].numel(), xs[1:].numel())
        jacobian_matrix_product = self._jmp(x, val, matrix, wrt=wrt)
        return jacobian_matrix_product.reshape(vs[0], *vs[1:], *vs[1:])

    def jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """jacobian matrix jacobian.T product - forward"""
        if wrt == "weight":
            raise NotImplementedError
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(x)
        else:
            assert matrix.shape == (xs[0], *xs[1:], *xs[1:])
        matrix = matrix.reshape(xs[0], xs[1:].numel(), xs[1:].numel())
        jacobian_matrix_jacobianT_product = self._jmjTp(
            x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=to_diag, diag_backprop=diag_backprop
        )
        return jacobian_matrix_jacobianT_product.reshape(vs[0], *vs[1:], *vs[1:])

    ### backward passes ###

    def vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """vector jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        assert vector.shape == vs
        vector = vector.reshape(vs[0], vs[1:].numel())
        vector_jacobian_product = self._vjp(x, val, vector, wrt=wrt)
        if wrt == "input":
            return vector_jacobian_product.reshape(xs[0], *xs[1:])
        elif wrt == "weight":
            return vector_jacobian_product

    def mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """matrix jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(val)
        else:
            assert matrix.shape == (vs[0], *vs[1:], *vs[1:])
        matrix = matrix.reshape(vs[0], vs[1:].numel(), vs[1:].numel())
        matrix_jacobian_product = self._mjp(x, val, matrix, wrt=wrt)
        if wrt == "input":
            return matrix_jacobian_product.reshape(xs[0], *xs[1:], *xs[1:])
        elif wrt == "weight":
            return matrix_jacobian_product

    def jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        """jacobian.T matrix jacobian product - backward"""
        if val is None:
            val = self.forward(x)
        xs, vs = x.shape, val.shape
        if matrix is None:
            matrix = identity(val)
        else:
            assert matrix.shape == (vs[0], *vs[1:], *vs[1:])
        matrix = matrix.reshape(vs[0], vs[1:].numel(), vs[1:].numel())
        jacobianT_matrix_jacobian_product = self._jTmjp(
            x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=to_diag, diag_backprop=diag_backprop
        )
        if wrt == "input":
            return jacobianT_matrix_jacobian_product.reshape(xs[0], *xs[1:], *xs[1:])
        elif wrt == "weight":
            return jacobianT_matrix_jacobian_product

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        """jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)"""
        assert x1.shape == x2.shape

    # slow implementations, to be overwritten by each module for efficient computation
    def _jvp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def _jmp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def _jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        if from_diag or to_diag or diag_backprop:
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bij,bjk,blk->bil", jacobian, matrix, jacobian)

    def _vjp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def _mjp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        if from_diag or to_diag or diag_backprop:
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)
        return torch.einsum("bji,bjk,bkl->bil", jacobian, matrix, jacobian)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        print(f"Ei! I'm doing this in the stupid way! ({self})")
        if from_diag or to_diag or diag_backprop:
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt).reshape(val.shape[0], val.shape[:1].numel(), -1)


class Sequential(nn.Sequential):
    """Subclass of sequential that also supports calculating the jacobian through an network"""

    def __init__(self, *args, add_hooks: bool = False):
        super().__init__(*args)
        self._modules_list = list(self._modules.values())

        self.add_hooks = add_hooks
        if self.add_hooks:
            self.feature_maps = []
            self.handles = []
            # def fw_hook(module, input, output):
            #    self.feature_maps.append(output.detach())
            for k in range(len(self._modules)):
                # self.handles.append(self._modules_list[k].register_forward_hook(fw_hook))
                self.handles.append(
                    self._modules_list[k].register_forward_hook(
                        lambda m, i, o: self.feature_maps.append(o.detach())
                    )
                )

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.add_hooks:
            self.feature_maps = [x]
        if not (jacobian is False):
            j = identity(x) if (not isinstance(jacobian, Tensor) and jacobian) else jacobian
        for module in self._modules.values():
            val = module(x)
            if not (jacobian is False):
                # j = module._jacobian_wrt_input_mult_left_vec(x, val, j)
                j = module._jmp(x, val, j)
            x = val
        if not (jacobian is False):
            return x, j
        return x

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        # forward pass for computing hook values
        if val is None:
            val = self.forward(x)

        # backward pass
        vs = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # print('layer:',self._modules_list[k], vector.shape)
            if wrt == "weight":
                v_k = self._modules_list[k]._vjp(
                    self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="weight"
                )
                if v_k is not None:
                    vs = [v_k] + vs
                if k == 0:
                    break
            vector = self._modules_list[k]._vjp(
                self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(vs, dim=1)
        elif wrt == "input":
            return vector

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        # forward pass
        if val is None:
            # val = self.forward(x)
            self.forward(x)
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            if wrt == "weight":
                m_k = self._modules_list[k]._mjp(
                    self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="weight"
                )
                if m_k is not None:
                    ms = [m_k] + ms
                if k == 0:
                    break
            matrix = self._modules_list[k]._mjp(
                self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(ms, dim=2)
        elif wrt == "input":
            return matrix

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product
        """
        # forward pass
        if val is None or val.shape[0] == 0:
            val = self.forward(x)
        if matrix is None or matrix.shape[0] == 0:
            matrix = torch.ones_like(val)
            from_diag = True
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # print('layer:',self._modules_list[k])
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    matrix,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    ms = [m_k] + ms
                    # ms = m_k + ms
                if k == 0:
                    break
            matrix = self._modules_list[k]._jTmjp(
                self.feature_maps[k],
                self.feature_maps[k + 1],
                matrix,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            if len(ms) == 0:
                return None  # case of a Sequential with no parametric layers inside
            # return ms
            if to_diag:
                return torch.cat(ms, dim=1)
            else:
                return tuple(ms)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape

        # forward passes
        self.forward(x1)
        feature_maps_1 = self.feature_maps
        self.forward(x2)
        feature_maps_2 = self.feature_maps

        if matrixes is None:
            matrixes = tuple(torch.ones_like(self.feature_maps[-1]) for _ in range(3))
            from_diag = True

        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # print('layer:',self._modules_list[k])
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp_batch2(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    matrixes,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    ms = m_k + ms
                if k == 0:
                    break
            matrixes = self._modules_list[k]._jTmjp_batch2(
                feature_maps_1[k],
                feature_maps_2[k],
                feature_maps_1[k + 1],
                feature_maps_2[k + 1],
                matrixes,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrixes
        elif wrt == "weight":
            if len(ms) == 0:
                return None  # case of a Sequential with no parametric layers
            return ms

    def _jacobian_sandwich(
        self,
        x: Tensor,
        tmp: Tensor,
        wrt="weight",
        diag_inp: bool = False,
        method="diagonal exact",
        tmp_is_identity=False,
    ):

        # forward pass for computing hook values
        z = self.forward(x)

        if tmp_is_identity:
            tmp = torch.ones_like(z)
            diag_inp = True

        # backward pass
        GGN = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            if method == "block exact":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else False
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich(
                        self.feature_maps[k],
                        self.feature_maps[k + 1],
                        tmp,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=False,
                    )
                    if GGN_k is not None:
                        GGN = [GGN_k] + GGN
                    if k == 0:
                        break
                tmp = self._modules_list[k]._jacobian_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=False,
                )
            elif method == "diagonal exact":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else False
                diag_out = True if k == 0 else False
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich(
                        self.feature_maps[k],
                        self.feature_maps[k + 1],
                        tmp,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=True,
                    )
                    if GGN_k is not None:
                        GGN = [GGN_k] + GGN
                    if k == 0:
                        break
                tmp = self._modules_list[k]._jacobian_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=diag_out,
                )
            elif method == "diagonal approx":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else True
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich(
                        self.feature_maps[k],
                        self.feature_maps[k + 1],
                        tmp,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=True,
                    )
                    if GGN_k is not None:
                        GGN = [GGN_k] + GGN
                    if k == 0:
                        break
                tmp = self._modules_list[k]._jacobian_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=True,
                )
            else:
                raise NotImplementedError
        if wrt == "input":
            GGN = tmp
        elif wrt == "weight":
            if method == "diagonal exact" or method == "diagonal approx":
                GGN = torch.cat(GGN, dim=1)
        return GGN

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        method="diagonal exact",
        tmp_is_identity=False,
    ):

        # x1 = x[indexes_1]
        # x2 = x[indexes_2]

        # forward passes for computing hook values
        self.forward(x1)
        feature_maps_1 = self.feature_maps

        z = self.forward(x2)
        feature_maps_2 = self.feature_maps

        if tmp_is_identity:
            tmps = tuple(torch.ones_like(z) for _ in range(3))
            diag_inp = True

        # backward pass
        GGN = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            if method == "block exact":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else False
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich_multipoint(
                        feature_maps_1[k],
                        feature_maps_2[k],
                        feature_maps_1[k + 1],
                        feature_maps_2[k + 1],
                        tmps,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=False,
                    )
                    if GGN_k[0] is not None:
                        GGN = [GGN_k[0] - GGN_k[1] - GGN_k[1].transpose(-2, -1) + GGN_k[2]] + GGN
                    if k == 0:
                        break
                tmps = self._modules_list[k]._jacobian_sandwich_multipoint(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    tmps,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=False,
                )
            elif method == "diagonal exact":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else False
                diag_out = True if k == 0 else False
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich_multipoint(
                        feature_maps_1[k],
                        feature_maps_2[k],
                        feature_maps_1[k + 1],
                        feature_maps_2[k + 1],
                        tmps,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=True,
                    )
                    if GGN_k[0] is not None:
                        GGN = [GGN_k[0] - 2 * GGN_k[1] + GGN_k[2]] + GGN
                    if k == 0:
                        break
                tmps = self._modules_list[k]._jacobian_sandwich_multipoint(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    tmps,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=diag_out,
                )
            elif method == "diagonal approx":
                diag_inp = diag_inp if k == len(self._modules_list) - 1 else True
                if wrt == "weight":
                    GGN_k = self._modules_list[k]._jacobian_sandwich_multipoint(
                        feature_maps_1[k],
                        feature_maps_2[k],
                        feature_maps_1[k + 1],
                        feature_maps_2[k + 1],
                        tmps,
                        wrt="weight",
                        diag_inp=diag_inp,
                        diag_out=True,
                    )
                    if GGN_k[0] is not None:
                        GGN = [GGN_k[0] - 2 * GGN_k[1] + GGN_k[2]] + GGN
                    if k == 0:
                        break
                tmps = self._modules_list[k]._jacobian_sandwich_multipoint(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    tmps,
                    wrt="input",
                    diag_inp=diag_inp,
                    diag_out=True,
                )
        if method == "diagonal exact" or method == "diagonal approx":
            if wrt == "weight":
                GGN = torch.cat(GGN, dim=1)
            elif wrt == "input":
                GGN = tmps[0] - 2 * tmps[1] + tmps[2]
            return GGN
        else:
            raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, *args, add_hooks: bool = False):
        super().__init__()

        self._F = Sequential(*args, add_hooks=add_hooks)

    def forward(self, x):
        return self._F(x) + x

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        vjp = self._F._vjp(x, None if val is None else val - x, vector, wrt=wrt)
        if wrt == "input":
            return vjp + vector
        elif wrt == "weight":
            return vjp

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        mjp = self._F._mjp(x, None if val is None else val - x, matrix, wrt=wrt)
        if wrt == "input":
            return matrix + mjp
        elif wrt == "weight":
            return mjp

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product
        """
        # TODO: deal with diagonal matrix
        if val is None:
            raise NotImplementedError
        if matrix is None:
            raise NotImplementedError
        jTmjp = self._F._jTmjp(
            x,
            None if val is None else val - x,
            matrix,
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            mjp = self._F._mjp(x, None if val is None else val - x, matrix, wrt=wrt)
            jTmp = self._F._mjp(
                x, None if val is None else val - x, matrix.transpose(1, 2), wrt=wrt
            ).transpose(1, 2)
            return jTmjp + mjp + jTmp + matrix
        elif wrt == "weight":
            return jTmjp

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        # TODO: deal with diagonal matrix
        if val1 is None:
            raise NotImplementedError
        if val2 is None:
            raise NotImplementedError
        if matrixes is None:
            raise NotImplementedError
        jTmjps = self._F._jTmjp_batch2(
            x1,
            x2,
            None if val1 is None else val1 - x1,
            None if val2 is None else val2 - x2,
            matrixes,
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            m11, m12, m22 = matrixes
            mjps = tuple(
                self._F._mjp(x_i, None if val_i is None else val_i - x_i, m, wrt=wrt)
                for x_i, val_i, m in [(x1, val1, m11), (x2, val2, m12), (x2, val2, m22)]
            )
            jTmps = tuple(
                self._F._mjp(
                    x_i, None if val_i is None else val_i - x_i, m.transpose(1, 2), wrt=wrt
                ).transpose(1, 2)
                for x_i, val_i, m in [(x1, val1, m11), (x1, val1, m12), (x2, val2, m22)]
            )
            # new_m11 = J1T * m11 * J1 + m11 * J1 + J1T * m11 + m11
            # new_m12 = J1T * m12 * J2 + m12 * J2 + J1T * m12 + m12
            # new_m22 = J2T * m22 * J2 + m22 * J2 + J2T * m22 + m22
            return tuple(jTmjp + mjp + jTmp + m for jTmjp, mjp, jTmp, m in zip(jTmjps, mjps, jTmps, matrixes))
        elif wrt == "weight":
            return jTmjps


class SkipConnection(nn.Module):
    def __init__(self, *args, add_hooks: bool = False):
        super().__init__()

        self._F = Sequential(*args, add_hooks=add_hooks)

    def forward(self, x):
        return torch.cat([x, self._F(x)], dim=1)

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        b, l = x.shape
        vjp = self._F._vjp(x, None if val is None else val[:, l:], vector[:, l:], wrt=wrt)
        if wrt == "input":
            return vector[:, :l] + vjp
        elif wrt == "weight":
            return vjp

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        b, l = x.shape
        mjp = self._F._mjp(x, None if val is None else val[:, l:], matrix[:, :, l:], wrt=wrt)
        if wrt == "input":
            return matrix[:, :, :l] + mjp
        elif wrt == "weight":
            return mjp

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        assert (not diag_backprop) or (diag_backprop and from_diag and to_diag)
        b, l = x.shape
        jTmjp = self._F._jTmjp(
            x,
            None if val is None else val[:, l:],
            matrix[:, l:, l:] if not from_diag else matrix[:, l:],
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if diag_backprop:
                return jTmjp + matrix[:, :l]
            mjp = self._F._mjp(x, None if val is None else val[:, l:], matrix[:, :l, l:], wrt=wrt)
            jTmp = self._F._mjp(
                x, None if val is None else val[:, l:], matrix[:, l:, :l].transpose(1, 2), wrt=wrt
            ).transpose(1, 2)
            return jTmjp + mjp + jTmp + matrix[:, :l, :l]
        elif wrt == "weight":
            return jTmjp

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        b, l = x1.shape
        # TODO: deal with diagonal matrix
        if val1 is None:
            raise NotImplementedError
        if val2 is None:
            raise NotImplementedError
        if matrixes is None:
            raise NotImplementedError
        jTmjps = self._F._jTmjp_batch2(
            x1,
            x2,
            None if val1 is None else val1[:, l:],
            None if val2 is None else val2[:, l:],
            tuple(m[:, l:, l:] for m in matrixes),
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            m11, m12, m22 = matrixes
            mjps = tuple(
                self._F._mjp(x_i, None if val_i is None else val_i[:, l:], m[:, :l, l:], wrt=wrt)
                for x_i, val_i, m in [(x1, val1, m11), (x2, val2, m12), (x2, val2, m22)]
            )
            jTmps = tuple(
                self._F._mjp(
                    x_i, None if val_i is None else val_i[:, l:], m[:, l:, :l].transpose(1, 2), wrt=wrt
                ).transpose(1, 2)
                for x_i, val_i, m in [(x1, val1, m11), (x1, val1, m12), (x2, val2, m22)]
            )
            # schematic of the update rule with jacobian products (neglecting batch size)
            # new_m11 = J1T * m11[l:,l:] * J1 + m11[l:,:l] * J1 + J1T * m11[:l,l:] + m11[:l,:l]
            # new_m12 = J1T * m12[l:,l:] * J2 + m12[l:,:l] * J2 + J1T * m12[:l,l:] + m12[:l,:l]
            # new_m22 = J2T * m22[l:,l:] * J2 + m22[l:,:l] * J2 + J2T * m22[:l,l:] + m22[:l,:l]
            return tuple(
                jTmjp + mjp + jTmp + m[:, :l, :l]
                for jTmjp, mjp, jTmp, m in zip(jTmjps, mjps, jTmps, matrixes)
            )
        elif wrt == "weight":
            return jTmjps


class Linear(AbstractJacobian, nn.Linear):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight, bias=None).movedim(-1, 1)

    def _jacobian_wrt_input_transpose_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight.T, bias=None).movedim(-1, 1)

    def get_jacobian(self, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            return self._jacobian_wrt_input(x, val)
        elif wrt == "weight":
            return self._jacobian_wrt_weight(x, val)

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        return self.weight

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1 = x.shape
        c2 = val.shape[1]
        out_identity = torch.diag_embed(torch.ones(c2, device=x.device))
        jacobian = torch.einsum("bk,ij->bijk", x, out_identity).reshape(b, c2, c2 * c1)
        if self.bias is not None:
            jacobian = torch.cat([jacobian, out_identity.unsqueeze(0).expand(b, -1, -1)], dim=2)
        return jacobian

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            return torch.einsum("kj,bj->bk", self.weight, vector)
        elif wrt == "weight":
            b, l = x.shape
            return torch.einsum("bkj,bj->bk", vector.view(b, l, l), x)

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return torch.einsum("bj,jk->bk", vector, self.weight)
        elif wrt == "weight":
            # jacobian = self._jacobian_wrt_weight(x,val)
            # return torch.einsum("bi,bij->bj", vector, jacobian)
            b, l = x.shape
            if self.bias is None:
                return torch.einsum("bi,bj->bij", vector, x).view(b, -1)
            else:
                return torch.cat([torch.einsum("bi,bj->bij", vector, x).view(b, -1), vector], dim=1)

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            return torch.einsum("kj,bji->bki", self.weight, matrix)
        elif wrt == "weight":
            jacobian = self._jacobian_wrt_weight(x, val)
            return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return torch.einsum("bij,jk->bik", matrix, self.weight)
        elif wrt == "weight":
            jacobian = self._jacobian_wrt_weight(x, val)
            return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        # if x.shape[0] == 0:
        if val is None or val.shape[0] == 0:
            val = self.forward(x)
        if matrix is None or matrix.shape[0] == 0:
            matrix = torch.ones_like(val)
            from_diag = True
        assert matrix.shape[0] != 0, f"{x.shape}"
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return torch.einsum("nm,bnj,jk->bmk", self.weight, matrix, self.weight)
            elif from_diag and not to_diag:
                # diag -> full
                return torch.einsum("nm,bn,nk->bmk", self.weight, matrix, self.weight)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("nm,bnj,jm->bm", self.weight, matrix, self.weight)
            elif from_diag and to_diag:
                # diag -> diag
                return torch.einsum("nm,bn,nm->bm", self.weight, matrix, self.weight)
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, matrix)
            elif not from_diag and to_diag:
                # full -> diag
                bs, _, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    print(torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1).shape)
                    return torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1)
                else:
                    return torch.cat(
                        [
                            torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1),
                            torch.einsum("bii->bi", matrix),
                        ],
                        dim=1,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                bs, _ = matrix.shape
                x_sq = x * x
                retu = torch.einsum("bj,bi->bij", x_sq, matrix)
                if self.bias is None:
                    print(retu.shape)
                    return retu
                else:
                    print(torch.cat([retu, matrix], dim=1).shape)
                    return torch.cat([retu, matrix], dim=1)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):
        if wrt == "weight":
            return [
                self._jTmjp_batch2_old(x1, x2, val1, val2, matrixes, wrt, from_diag, to_diag, diag_backprop)
            ]
        else:
            return self._jTmjp_batch2_old(
                x1, x2, val1, val2, matrixes, wrt, from_diag, to_diag, diag_backprop
            )

    def _jTmjp_batch2_old(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(torch.einsum("nm,bnj,jk->bmk", self.weight, m, self.weight) for m in matrixes)
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(
                    torch.einsum("nm,bn,nk->bmk", self.weight, m_diag, self.weight) for m_diag in matrixes
                )
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(torch.einsum("nm,bnj,jm->bm", self.weight, m, self.weight) for m in matrixes)
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("nm,bn,nm->bm", self.weight, m_diag, self.weight) for m_diag in matrixes
                )
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                m11, m12, m22 = matrixes
                jac_1 = self._jacobian_wrt_weight(x1, val1)
                jac_2 = self._jacobian_wrt_weight(x2, val2)
                return tuple(
                    torch.einsum("bji,bjk,bkq->biq", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and not to_diag:
                # diag -> full
                m11, m12, m22 = matrixes
                jac_1 = self._jacobian_wrt_weight(x1, val1)
                jac_2 = self._jacobian_wrt_weight(x2, val2)
                return tuple(
                    torch.einsum("bji,bj,bjk->bik", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif not from_diag and to_diag:
                # full -> diag
                m11, m12, m22 = matrixes
                bs, _ = x1.shape
                if self.bias is None:
                    return tuple(
                        torch.einsum("bj,bii,bj->bij", x_i, m, x_j).view(bs, -1)
                        for x_i, m, x_j in [(x1, m11, x1), (x1, m12, x2), (x2, m22, x2)]
                    )
                else:
                    return tuple(
                        torch.cat(
                            [
                                torch.einsum("bj,bii,bj->bij", x_i, m, x_j).view(bs, -1),
                                torch.einsum("bii->bi", m),
                            ],
                            dim=1,
                        )
                        for x_i, m, x_j in [(x1, m11, x1), (x1, m12, x2), (x2, m22, x2)]
                    )
            elif from_diag and to_diag:
                # diag -> diag
                m11_diag, m12_diag, m22_diag = matrixes
                bs, _ = x1.shape
                if self.bias is None:
                    return tuple(
                        torch.einsum("bj,bi,bj->bij", x_i, m_diag, x_j).view(bs, -1)
                        for x_i, m_diag, x_j in [(x1, m11_diag, x1), (x1, m12_diag, x2), (x2, m22_diag, x2)]
                    )
                else:
                    return tuple(
                        torch.cat(
                            [torch.einsum("bj,bi,bj->bij", x_i, m_diag, x_j).view(bs, -1), m_diag], dim=1
                        )
                        for x_i, m_diag, x_j in [(x1, m11_diag, x1), (x1, m12_diag, x2), (x2, m22_diag, x2)]
                    )

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp, wrt="input", diag_inp: bool = True, diag_out: bool = True
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return torch.einsum("nm,bnj,jk->bmk", self.weight, tmp, self.weight)
            elif diag_inp and not diag_out:
                # diag -> full
                return torch.einsum("nm,bn,nk->bmk", self.weight, tmp, self.weight)
            elif not diag_inp and diag_out:
                # full -> diag
                return torch.einsum("nm,bnj,jm->bm", self.weight, tmp, self.weight)
            elif diag_inp and diag_out:
                # diag -> diag
                return torch.einsum("nm,bn,nm->bm", self.weight, tmp, self.weight)
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                # TODO: implement more effciently
                return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                # TODO: implement more effciently
                return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
            elif not diag_inp and diag_out:
                # full -> diag
                bs, _, _ = tmp.shape
                if self.bias is None:
                    return torch.einsum("bj,bii,bj->bij", x, tmp, x).view(bs, -1)
                else:
                    return torch.cat(
                        [
                            torch.einsum("bj,bii,bj->bij", x, tmp, x).view(bs, -1),
                            torch.einsum("bii->bi", tmp),
                        ],
                        dim=1,
                    )
            elif diag_inp and diag_out:
                # diag -> diag
                bs, _ = tmp.shape
                if self.bias is None:
                    return torch.einsum("bj,bi,bj->bij", x, tmp, x).view(bs, -1)
                else:
                    return torch.cat([torch.einsum("bj,bi,bj->bij", x, tmp, x).view(bs, -1), tmp], dim=1)

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                return tuple(torch.einsum("nm,bnj,jk->bmk", self.weight, tmp, self.weight) for tmp in tmps)
            elif diag_inp and not diag_out:
                return tuple(
                    torch.einsum("nm,bn,nk->bmk", self.weight, tmp_diag, self.weight) for tmp_diag in tmps
                )
            elif not diag_inp and diag_out:
                # full -> diag
                return tuple(torch.einsum("nm,bnj,jm->bm", self.weight, tmp, self.weight) for tmp in tmps)
            elif diag_inp and diag_out:
                # diag -> diag
                return tuple(
                    torch.einsum("nm,bn,nm->bm", self.weight, tmp_diag, self.weight) for tmp_diag in tmps
                )
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                raise NotImplementedError
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                tmp11, tmp12, tmp22 = tmps
                bs, _, _ = tmp11.shape
                if self.bias is None:
                    diag_tmp11 = torch.einsum("bj,bii,bj->bij", x1, tmp11, x1).view(bs, -1)
                    diag_tmp12 = torch.einsum("bj,bii,bj->bij", x1, tmp12, x2).view(bs, -1)
                    diag_tmp22 = torch.einsum("bj,bii,bj->bij", x2, tmp22, x2).view(bs, -1)
                else:
                    diag_tmp11 = torch.cat(
                        [
                            torch.einsum("bj,bii,bj->bij", x1, tmp11, x1).view(bs, -1),
                            torch.einsum("bii->bi", tmp11),
                        ],
                        dim=1,
                    )
                    diag_tmp12 = torch.cat(
                        [
                            torch.einsum("bj,bii,bj->bij", x1, tmp12, x2).view(bs, -1),
                            torch.einsum("bii->bi", tmp12),
                        ],
                        dim=1,
                    )
                    diag_tmp22 = torch.cat(
                        [
                            torch.einsum("bj,bii,bj->bij", x2, tmp22, x2).view(bs, -1),
                            torch.einsum("bii->bi", tmp22),
                        ],
                        dim=1,
                    )
                return (diag_tmp11, diag_tmp12, diag_tmp22)
            elif diag_inp and diag_out:
                # diag -> diag
                diag_tmp11, diag_tmp12, diag_tmp22 = tmps
                bs, _ = diag_tmp11.shape
                if self.bias is None:
                    diag_tmp11 = torch.einsum("bj,bi,bj->bij", x1, diag_tmp11, x1).view(bs, -1)
                    diag_tmp12 = torch.einsum("bj,bi,bj->bij", x1, diag_tmp12, x2).view(bs, -1)
                    diag_tmp22 = torch.einsum("bj,bi,bj->bij", x2, diag_tmp22, x2).view(bs, -1)
                else:
                    diag_tmp11 = torch.cat(
                        [torch.einsum("bj,bi,bj->bij", x1, diag_tmp11, x1).view(bs, -1), diag_tmp11], dim=1
                    )
                    diag_tmp12 = torch.cat(
                        [torch.einsum("bj,bi,bj->bij", x1, diag_tmp12, x2).view(bs, -1), diag_tmp12], dim=1
                    )
                    diag_tmp22 = torch.cat(
                        [torch.einsum("bj,bi,bj->bij", x2, diag_tmp22, x2).view(bs, -1), diag_tmp22], dim=1
                    )
                return (diag_tmp11, diag_tmp12, diag_tmp22)

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jk->bmk", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jm->bm", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nk->bmk", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nm->bm", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bjk,bkq->biq", jacobian, tmp, jacobian)

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        tmp_diag = torch.diagonal(tmp, dim1=1, dim2=2)
        return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp_diag)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bj,bjq->biq", jacobian, tmp_diag, jacobian)

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1 = x.shape
        c2 = val.shape[1]

        Jt_tmp_J = torch.bmm(tmp_diag.unsqueeze(2), (x**2).unsqueeze(1)).view(b, c1 * c2)

        if self.bias is not None:
            Jt_tmp_J = torch.cat([Jt_tmp_J, tmp_diag], dim=1)

        return Jt_tmp_J


class PosLinear(AbstractJacobian, nn.Linear):
    def forward(self, x: Tensor):
        bias = F.softplus(self.bias) if self.bias is not None else self.bias
        val = F.linear(x, F.softplus(self.weight), bias)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), F.softplus(self.weight), bias=None).movedim(-1, 1)


class L2Norm(nn.Module):
    """L2 normalization layer"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def get_jacobian(self, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            return self._jacobian_wrt_input(x, val)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, d = x.shape
        norm = torch.norm(x, p=2, dim=1)

        normalized_x = torch.einsum("b,bi->bi", 1 / (norm + 1e-6), x)

        jacobian = torch.einsum("bi,bj->bij", normalized_x, normalized_x)
        jacobian = torch.diag(torch.ones(d, device=x.device)).expand(b, d, d) - jacobian
        jacobian = torch.einsum("b,bij->bij", 1 / (norm + 1e-6), jacobian)

        return jacobian

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            jacobian = self._jacobian_wrt_input(x, val)
            return torch.einsum("bij,bj->bi", jacobian, vector)
        elif wrt == "weight":
            return None

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            jacobian = self._jacobian_wrt_input(x, val)
            return torch.einsum("bi,bij->bj", vector, jacobian)
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            jacobian = self._jacobian_wrt_input(x, val)
            if matrix is None:
                return jacobian
            return torch.einsum("bij,bjk->bik", jacobian, matrix)
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            jacobian = self._jacobian_wrt_input(x, val)
            if matrix is None:
                return jacobian
            return torch.einsum("bij,bjk->bik", matrix, jacobian)
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bik,bkl->bjl", jacobian, matrix, jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bi,bil->bjl", jacobian, matrix, jacobian)
            elif not from_diag and to_diag:
                # full -> diag
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bik,bkj->bj", jacobian, matrix, jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bi,bij->bj", jacobian, matrix, jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            m11, m12, m22 = matrixes
            jac_1 = self.get_jacobian(x1, val1, wrt=wrt)
            jac_2 = self.get_jacobian(x2, val2, wrt=wrt)

            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    torch.einsum("bji,bjk,bkq->biq", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(
                    torch.einsum("bji,bj,bjk->bik", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(
                    torch.einsum("bji,bjk,bki->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("bji,bj,bji->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bik,bkl->bjl", jacobian, tmp, jacobian)
            elif diag_inp and not diag_out:
                # diag -> full
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bi,bil->bjl", jacobian, tmp, jacobian)
            elif not diag_inp and diag_out:
                # full -> diag
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bik,bkj->bj", jacobian, tmp, jacobian)
            elif diag_inp and diag_out:
                # diag -> diag
                jacobian = self._jacobian_wrt_input(x, val)
                return torch.einsum("bij,bi,bij->bj", jacobian, tmp, jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            tmp11, tmp12, tmp22 = tmps
            if not diag_inp and not diag_out:
                # full -> full
                jacobian_1 = self.get_jacobian(x1, val1, wrt=wrt)
                jacobian_2 = self.get_jacobian(x2, val2, wrt=wrt)
                tmp11 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_1, tmp11, jacobian_1)
                tmp12 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_1, tmp12, jacobian_2)
                tmp22 = torch.einsum("bnm,bnj,bjk->bmk", jacobian_2, tmp22, jacobian_2)
            elif diag_inp and not diag_out:
                # diag -> full
                jacobian_1 = self.get_jacobian(x1, val1, wrt=wrt)
                jacobian_2 = self.get_jacobian(x2, val2, wrt=wrt)
                tmp11 = torch.einsum("bnm,bn,bnk->bmk", jacobian_1, tmp11, jacobian_1)
                tmp12 = torch.einsum("bnm,bn,bnk->bmk", jacobian_1, tmp12, jacobian_2)
                tmp22 = torch.einsum("bnm,bn,bnk->bmk", jacobian_2, tmp22, jacobian_2)
            elif not diag_inp and diag_out:
                # full -> diag
                jacobian_1 = self.get_jacobian(x1, val1, wrt=wrt)
                jacobian_2 = self.get_jacobian(x2, val2, wrt=wrt)
                tmp11 = torch.einsum("bij,bik,bkj->bj", jacobian_1, tmp11, jacobian_1)
                tmp12 = torch.einsum("bij,bik,bkj->bj", jacobian_1, tmp12, jacobian_2)
                tmp22 = torch.einsum("bij,bik,bkj->bj", jacobian_2, tmp22, jacobian_2)
            elif diag_inp and diag_out:
                # diag -> diag
                jacobian_1 = self.get_jacobian(x1, val1, wrt=wrt)
                jacobian_2 = self.get_jacobian(x2, val2, wrt=wrt)
                tmp11 = torch.einsum("bij,bi,bij->bj", jacobian_1, tmp11, jacobian_1)
                tmp12 = torch.einsum("bij,bi,bij->bj", jacobian_1, tmp12, jacobian_2)
                tmp22 = torch.einsum("bij,bi,bij->bj", jacobian_2, tmp22, jacobian_2)
            return (tmp11, tmp12, tmp22)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_input(x, val)
        return torch.einsum("bij,bik,bkl->bjl", jacobian, tmp, jacobian)

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        else:
            raise NotImplementedError

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


class Upsample(AbstractJacobian, nn.Upsample):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        xs = x.shape
        vs = val.shape

        dims1 = tuple(range(1, x.ndim))
        dims2 = tuple(range(-x.ndim + 1, 0))

        return (
            F.interpolate(
                jac_in.movedim(dims1, dims2).reshape(-1, *xs[1:]),
                self.size,
                self.scale_factor,
                self.mode,
                self.align_corners,
            )
            .reshape(xs[0], *jac_in.shape[x.ndim :], *vs[1:])
            .movedim(dims2, dims1)
        )

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2
            assert matrix.shape == (b, c2 * h2 * w2, c2 * h2 * w2)

            weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

            matrix = matrix.reshape(b, c2, h2 * w2, c2, h2 * w2)
            matrix = matrix.movedim(2, 3)
            matrix_J = F.conv2d(
                matrix.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
                weight=weight,
                bias=None,
                stride=int(self.scale_factor),
                padding=0,
                dilation=1,
                groups=1,
            ).reshape(b * c2, c2, h2 * w2, h1 * w1)

            matrix_J = matrix_J.movedim(2, 3)
            return matrix_J.reshape(b, c2 * h2 * w2, c1 * h1 * w1)
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, m) for m in matrixes)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, m) for m in matrixes)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tuple(self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, tmp) for tmp in tmps)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return tuple(self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, tmp) for tmp in tmps)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        assert c1 == c2

        weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp = tmp.reshape(b, c2, h2 * w2, c2, h2 * w2)
        tmp = tmp.movedim(2, 3)
        tmp_J = F.conv2d(
            tmp.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h2 * w2, h1 * w1)

        Jt_tmpt = tmp_J.movedim(-1, -2)

        Jt_tmpt_J = F.conv2d(
            Jt_tmpt.reshape(b * c2 * c2 * h1 * w1, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h1 * w1, h1 * w1)

        Jt_tmp_J = Jt_tmpt_J.movedim(-1, -2)

        Jt_tmp_J = Jt_tmp_J.reshape(b, c2, c2, h1 * w1, h1 * w1)
        Jt_tmp_J = Jt_tmp_J.movedim(2, 3)
        Jt_tmp_J = Jt_tmp_J.reshape(b, c2 * h1 * w1, c2 * h1 * w1)

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        weight = torch.ones(c2, c1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp_diag = F.conv2d(
            tmp_diag.reshape(-1, c2, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        )

        return tmp_diag.reshape(b, c1 * h1 * w1)


class Conv1d(AbstractJacobian, nn.Conv1d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return (
            F.conv1d(
                jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[3:], c2, l2)
            .movedim((-2, -1), (1, 2))
        )


class ConvTranspose1d(AbstractJacobian, nn.ConvTranspose1d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return (
            F.conv_transpose1d(
                jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[3:], c2, l2)
            .movedim((-2, -1), (1, 2))
        )


def compute_reversed_padding(padding, kernel_size=1):
    return kernel_size - 1 - padding


class Conv2d(AbstractJacobian, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        dw_padding_h = compute_reversed_padding(self.padding[0], kernel_size=self.kernel_size[0])
        dw_padding_w = compute_reversed_padding(self.padding[1], kernel_size=self.kernel_size[1])
        self.dw_padding = (dw_padding_h, dw_padding_w)

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        return (
            F.conv2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        output_identity = torch.eye(c1 * h1 * w1).unsqueeze(0).expand(b, -1, -1)
        output_identity = output_identity.reshape(b, c1, h1, w1, c1 * h1 * w1)

        # convolve each column
        jacobian = self._jacobian_wrt_input_mult_left_vec(x, val, output_identity)

        # reshape as a (num of output)x(num of input) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c1 * h1 * w1)

        return jacobian

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        output_identity = torch.eye(c2 * c1 * kernel_h * kernel_w)
        # expand rows as [(input channels)x(kernel height)x(kernel width)] cubes, one for each output channel
        output_identity = output_identity.reshape(c2, c1, kernel_h, kernel_w, c2 * c1 * kernel_h * kernel_w)

        reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

        # convolve each base element and compute the jacobian
        jacobian = (
            F.conv_transpose2d(
                output_identity.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, kernel_h, kernel_w),
                weight=reversed_inputs,
                bias=None,
                stride=self.stride,
                padding=self.dw_padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(c2, *output_identity.shape[4:], b, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # transpose the result in (output height)x(output width)
        jacobian = torch.flip(jacobian, [-3, -2])
        # switch batch size and output channel
        jacobian = jacobian.movedim(0, 1)
        # reshape as a (num of output)x(num of weights) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c2 * c1 * kernel_h * kernel_w)
        return jacobian

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return self._jacobian_wrt_input_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
            else:
                b_term = torch.einsum("bchw->bc", vector.reshape(val.shape))
                return torch.cat(
                    [self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1), b_term],
                    dim=1,
                )

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return self._jacobian_wrt_input_mult_left(x, val, matrix)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, matrix)
            else:
                b, c, h, w = val.shape
                b_term = torch.einsum("bvchw->bvc", matrix.reshape(b, -1, c, h, w))
                return torch.cat([self._jacobian_wrt_weight_mult_left(x, val, matrix), b_term], dim=2)

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, matrix)
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                if self.bias is None:
                    return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, matrix)
                else:
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    return matrix
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                if self.bias is None:
                    return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, matrix)
                else:
                    # TODO: Implement this in a smarter way
                    return torch.diagonal(
                        self._jTmjp(x, val, matrix, wrt=wrt, from_diag=from_diag, to_diag=False),
                        dim1=1,
                        dim2=2,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, matrix)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):
        if wrt == "weight":
            return [
                self._jTmjp_batch2_old(x1, x2, val1, val2, matrixes, wrt, from_diag, to_diag, diag_backprop)
            ]
        else:
            return self._jTmjp_batch2_old(
                x1, x2, val1, val2, matrixes, wrt, from_diag, to_diag, diag_backprop
            )

    def _jTmjp_batch2_old(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, m) for m in matrixes
                )  # not dependent on x1,val1, only on their shape
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, m) for m in matrixes
                )  # not dependent on x1,val1, only on their shape
        elif wrt == "weight":
            m11, m12, m22 = matrixes
            if not from_diag and not to_diag:
                # full -> full
                if self.bias is None:
                    return tuple(
                        self._jacobian_wrt_weight_T_mult_right(
                            x_i, val_i, self._jacobian_wrt_weight_mult_left(x_j, val_j, m)
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
                else:
                    return tuple(
                        self._mjp(
                            x_i, val_i, self._mjp(x_j, val_j, m, wrt=wrt).movedim(-2, -1), wrt=wrt
                        ).movedim(-2, -1)
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                # TODO: Implement this in a smarter way
                if self.bias is None:
                    return tuple(
                        torch.diagonal(
                            self._jacobian_wrt_weight_T_mult_right(
                                x_i, val_i, self._jacobian_wrt_weight_mult_left(x_j, val_j, m)
                            ),
                            dim1=1,
                            dim2=2,
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
                else:
                    return tuple(
                        torch.diagonal(
                            self._mjp(
                                x_i, val_i, self._mjp(x_j, val_j, m, wrt=wrt).movedim(-2, -1), wrt=wrt
                            ).movedim(-2, -1),
                            dim1=1,
                            dim2=2,
                        )
                        for x_i, val_i, m, x_j, val_j in [
                            (x1, val1, m11, x1, val1),
                            (x1, val1, m12, x2, val2),
                            (x2, val2, m22, x2, val2),
                        ]
                    )
            elif from_diag and to_diag:
                # diag -> diag
                if self.bias is None:
                    return tuple(
                        (
                            self._jacobian_wrt_weight_sandwich_diag_to_diag(x1, val1, m11),
                            self._jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
                                x1, x2, val1, val2, m12
                            ),
                            self._jacobian_wrt_weight_sandwich_diag_to_diag(x2, val2, m22),
                        )
                    )
                else:
                    raise NotImplementedError

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp, wrt="input", diag_inp: bool = True, diag_out: bool = True
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tuple(
                    self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, tmp_diag) for tmp_diag in tmps
                )  # not dependent on x1,val1, only on their shape
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return tuple(
                    self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, tmp_diag) for tmp_diag in tmps
                )  # not dependent on x1,val1, only on their shape
        elif wrt == "weight":
            if not diag_inp and not diag_out:
                # full -> full
                tmp11, tmp12, tmp22 = tmps
                tmp11 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x1, val1, tmp11)
                )
                tmp12 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x2, val2, tmp12)
                )
                tmp22 = self._jacobian_wrt_weight_T_mult_right(
                    x2, val2, self._jacobian_wrt_weight_mult_left(x2, val2, tmp22)
                )
                return (tmp11, tmp12, tmp22)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                # TODO: Implement this in a smarter way
                tmp11, tmp12, tmp22 = tmps
                tmp11 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x1, val1, tmp11)
                )
                tmp12 = self._jacobian_wrt_weight_T_mult_right(
                    x1, val1, self._jacobian_wrt_weight_mult_left(x2, val2, tmp12)
                )
                tmp22 = self._jacobian_wrt_weight_T_mult_right(
                    x2, val2, self._jacobian_wrt_weight_mult_left(x2, val2, tmp22)
                )
                return tuple(torch.diagonal(tmp, dim1=1, dim2=2) for tmp in [tmp11, tmp12, tmp22])
            elif diag_inp and diag_out:
                # diag -> diag
                diag_tmp11, diag_tmp12, diag_tmp22 = tmps
                diag_tmp11 = self._jacobian_wrt_weight_sandwich_diag_to_diag(x1, val1, diag_tmp11)
                diag_tmp12 = self._jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
                    x1, val1, x2, val2, diag_tmp12
                )
                diag_tmp22 = self._jacobian_wrt_weight_sandwich_diag_to_diag(x2, val2, diag_tmp22)
                return (diag_tmp11, diag_tmp12, diag_tmp22)

    def _jacobian_wrt_input_T_mult_right(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_cols = tmp.shape[-1]
        assert list(tmp.shape) == [b, c2 * h2 * w2, num_of_cols]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)

        # convolve each column
        Jt_tmp = (
            F.conv_transpose2d(
                tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of column) matrix, one for each batch size
        Jt_tmp = Jt_tmp.reshape(b, c1 * h1 * w1, num_of_cols)
        return Jt_tmp

    def _jacobian_wrt_input_mult_left(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_rows = tmp.shape[-2]
        assert list(tmp.shape) == [b, num_of_rows, c2 * h2 * w2]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows

        # convolve each column
        Jt_tmptt_cols = (
            F.conv_transpose2d(
                tmpt_cols.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmpt_cols.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of output) matrix, one for each batch size
        Jt_tmptt_cols = Jt_tmptt_cols.reshape(b, c1 * h1 * w1, num_of_rows)

        # transpose
        tmp_J = Jt_tmptt_cols.movedim(1, 2)
        return tmp_J

    def _jacobian_wrt_weight_T_mult_right(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        num_of_cols = tmp.shape[-1]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)
        # transpose the images in (output height)x(output width)
        tmp = torch.flip(tmp, [-3, -2])
        # switch batch size and output channel
        tmp = tmp.movedim(0, 1)

        if use_less_memory:
            # define moving sum for Jt_tmp
            Jt_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_cols, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmp[:, i : i + 1, :, :, :]

                # convolve each column
                Jt_tmp_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                Jt_tmp_single_batch = Jt_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)
                Jt_tmp[i, :, :] = Jt_tmp_single_batch

        else:
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmp = (
                F.conv2d(
                    tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of weights)x(num of column) matrix
            Jt_tmp = Jt_tmp.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)

        return Jt_tmp

    def _jacobian_wrt_weight_mult_left(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        kernel_h, kernel_w = self.kernel_size
        num_of_rows = tmp.shape[-2]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows
        # transpose the images in (output height)x(output width)
        tmpt_cols = torch.flip(tmpt_cols, [-3, -2])
        # switch batch size and output channel
        tmpt_cols = tmpt_cols.movedim(0, 1)

        if use_less_memory:

            tmp_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_rows, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmpt_cols[:, i : i + 1, :, :, :]

                # convolve each column
                tmp_J_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                tmp_J_single_batch = tmp_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
                tmp_J[i, :, :] = tmp_J_single_batch

            # transpose
            tmp_J = tmp_J.movedim(-1, -2)
        else:
            # set the weight to the convolution
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmptt_cols = (
                F.conv2d(
                    tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of input)x(num of output) matrix, one for each batch size
            Jt_tmptt_cols = Jt_tmptt_cols.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
            # transpose
            tmp_J = Jt_tmptt_cols.movedim(0, 1)

        return tmp_J

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_input_mult_left(x, val, self._jacobian_wrt_input_T_mult_right(x, val, tmp))

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)

        output_tmp = (
            F.conv_transpose2d(
                input_tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight**2,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(b, *input_tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        diag_Jt_tmp_J = output_tmp.reshape(b, c1 * h1 * w1)
        return diag_Jt_tmp_J

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_weight_mult_left(
            x, val, self._jacobian_wrt_weight_T_mult_right(x, val, tmp)
        )

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        # TODO: Implement this in a smarter way
        return torch.diagonal(self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp), dim1=1, dim2=2)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        _, _, kernel_h, kernel_w = self.weight.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)
        # transpose the images in (output height)x(output width)
        input_tmp = torch.flip(input_tmp, [-3, -2, -1])
        # switch batch size and output channel
        input_tmp = input_tmp.movedim(0, 1)

        # define moving sum for Jt_tmp
        output_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
        flip_squared_input = torch.flip(x, [-3, -2, -1]).movedim(0, 1) ** 2

        for i in range(b):
            # set the weight to the convolution
            weigth_sq = flip_squared_input[:, i : i + 1, :, :]
            input_tmp_single_batch = input_tmp[:, i : i + 1, :, :]

            output_tmp_single_batch = (
                F.conv2d(
                    input_tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                    weight=weigth_sq,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *input_tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            output_tmp_single_batch = torch.flip(output_tmp_single_batch, [-4, -3])
            # reshape as a (num of weights)x(num of column) matrix
            output_tmp_single_batch = output_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
            output_tmp[i, :] = output_tmp_single_batch

        if self.bias is not None:
            bias_term = tmp_diag.reshape(b, c2, h2 * w2)
            bias_term = torch.sum(bias_term, 2)
            output_tmp = torch.cat([output_tmp, bias_term], dim=1)

        return output_tmp

    def _jacobian_wrt_weight_sandwich_diag_to_diag_multipoint(
        self, x: Tensor, xB: Tensor, val: Tensor, valB: Tensor, tmp_diag: Tensor
    ) -> Tensor:

        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        _, _, kernel_h, kernel_w = self.weight.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)
        # transpose the images in (output height)x(output width)
        input_tmp = torch.flip(input_tmp, [-3, -2, -1])
        # switch batch size and output channel
        input_tmp = input_tmp.movedim(0, 1)

        # define moving sum for Jt_tmp
        output_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
        flip_input = torch.flip(x, [-3, -2, -1]).movedim(0, 1)
        flip_inputB = torch.flip(xB, [-3, -2, -1]).movedim(0, 1)
        flip_squared_input = flip_input * flip_inputB

        for i in range(b):
            # set the weight to the convolution
            weigth_sq = flip_squared_input[:, i : i + 1, :, :]
            input_tmp_single_batch = input_tmp[:, i : i + 1, :, :]

            output_tmp_single_batch = (
                F.conv2d(
                    input_tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                    weight=weigth_sq,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *input_tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            output_tmp_single_batch = torch.flip(output_tmp_single_batch, [-4, -3])
            # reshape as a (num of weights)x(num of column) matrix
            output_tmp_single_batch = output_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
            output_tmp[i, :] = output_tmp_single_batch

        if self.bias is not None:
            bias_term = tmp_diag.reshape(b, c2, h2 * w2)
            bias_term = torch.sum(bias_term, 2)
            output_tmp = torch.cat([output_tmp, bias_term], dim=1)

        return output_tmp


class ConvTranspose2d(AbstractJacobian, nn.ConvTranspose2d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        return (
            F.conv_transpose2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )


class Conv3d(AbstractJacobian, nn.Conv3d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return (
            F.conv3d(
                jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[5:], c2, d2, h2, w2)
            .movedim((-4, -3, -2, -1), (1, 2, 3, 4))
        )


class ConvTranspose3d(AbstractJacobian, nn.ConvTranspose3d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return (
            F.conv_transpose3d(
                jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[5:], c2, d2, h2, w2)
            .movedim((-4, -3, -2, -1), (1, 2, 3, 4))
        )


class Reshape(AbstractJacobian, nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], *self.dims)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in.reshape(jac_in.shape[0], *self.dims, *jac_in.shape[2:])

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrix
            elif from_diag and not to_diag:
                # diag -> full
                return torch.diag_embed(matrix)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.diagonal(matrix, dim1=1, dim2=2)
            elif from_diag and to_diag:
                # diag -> diag
                return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrixes
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(torch.diag_embed(m) for m in matrixes)
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(torch.diagonal(m, dim1=1, dim2=2) for m in matrixes)
            elif from_diag and to_diag:
                # diag -> diag
                return matrixes
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tmp
            elif diag_inp and not diag_out:
                # diag -> full
                return torch.diag_embed(tmp)
            elif not diag_inp and diag_out:
                # full -> diag
                return torch.diagonal(tmp, dim1=1, dim2=2)
            elif diag_inp and diag_out:
                # diag -> diag
                return tmp
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tmps
            elif diag_inp and not diag_out:
                # diag -> full
                return tuple(torch.diag_embed(tmp) for tmp in tmps)
            elif not diag_inp and diag_out:
                # full -> diag
                return tuple(torch.diagonal(tmp, dim1=1, dim2=2) for tmp in tmps)
            elif diag_inp and diag_out:
                # diag -> diag
                return tmps
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return tmp
        elif not diag_inp and diag_out:
            return torch.diagonal(tmp, dim1=1, dim2=2)
        elif diag_inp and not diag_out:
            return torch.diag_embed(tmp)
        elif diag_inp and diag_out:
            return tmp

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


class Flatten(AbstractJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], -1)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        if jac_in.ndim == 5:  # 1d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[3:])
        if jac_in.ndim == 7:  # 2d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[4:])
        if jac_in.ndim == 9:  # 3d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[5:])

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrix
            elif from_diag and not to_diag:
                # diag -> full
                return torch.diag_embed(matrix)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.diagonal(matrix, dim1=1, dim2=2)
            elif from_diag and to_diag:
                # diag -> diag
                return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrixes
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(torch.diag_embed(m) for m in matrixes)
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(torch.diagonal(m, dim1=1, dim2=2) for m in matrixes)
            elif from_diag and to_diag:
                # diag -> diag
                return matrixes
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tmp
            elif diag_inp and not diag_out:
                # diag -> full
                return torch.diag_embed(tmp)
            elif not diag_inp and diag_out:
                # full -> diag
                return torch.diagonal(tmp, dim1=1, dim2=2)
            elif diag_inp and diag_out:
                # diag -> diag
                return tmp
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tmps
            elif diag_inp and not diag_out:
                # diag -> full
                return tuple(torch.diag_embed(tmp) for tmp in tmps)
            elif not diag_inp and diag_out:
                # full -> diag
                return tuple(torch.diagonal(tmp, dim1=1, dim2=2) for tmp in tmps)
            elif diag_inp and diag_out:
                # diag -> diag
                return tmps
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return tmp
        elif not diag_inp and diag_out:
            return torch.diagonal(tmp, dim1=1, dim2=2)
        elif diag_inp and not diag_out:
            return torch.diag_embed(tmp)
        elif diag_inp and diag_out:
            return tmp

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


class AbstractActivationJacobian:
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        return jac_in * jac.reshape(jac.shape + (1,) * n)

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class Softmax(AbstractActivationJacobian, nn.Softmax):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        if self.dim == 0:
            raise ValueError("Jacobian computation not supported for `dim=0`")
        jac = torch.diag_embed(val) - torch.matmul(val.unsqueeze(-1), val.unsqueeze(-2))
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        jac = jac.reshape((1,) * n + jac.shape)
        if jac_in.ndim == 4:
            return (jac @ jac_in.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        if jac_in.ndim == 5:
            return (jac @ jac_in.permute(3, 4, 0, 1, 2)).permute(2, 3, 4, 0, 1)
        if jac_in.ndim == 6:
            return (jac @ jac_in.permute(3, 4, 5, 0, 1, 2)).permute(3, 4, 5, 0, 1, 2)
        return jac @ jac_in


class BatchNorm1d(AbstractActivationJacobian, nn.BatchNorm1d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class BatchNorm2d(AbstractActivationJacobian, nn.BatchNorm2d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class BatchNorm3d(AbstractActivationJacobian, nn.BatchNorm3d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class MaxPool1d(AbstractJacobian, nn.MaxPool1d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, l1, *jac_in_orig_shape[3:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), l2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :].reshape(*val.shape, *jac_in_orig_shape[3:])
        return jac_in


class MaxPool2d(AbstractJacobian, nn.MaxPool2d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, h1 * w1, *jac_in_orig_shape[4:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :, :].reshape(*val.shape, *jac_in_orig_shape[4:])
        return jac_in

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            return None

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2

            matrix = (
                matrix.reshape(b, c1, h2 * w2, c1, h2 * w2)
                .movedim(-2, -3)
                .reshape(b * c1 * c1, h2 * w2, h2 * w2)
            )
            # indexes for batch, channel and row
            arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1 * h2 * w2), h2 * w2).long()
            arange_repeated = arange_repeated.reshape(b * c1 * c1 * h2 * w2, h2 * w2)
            # indexes for col
            idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
            idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1 * h2 * w2, h2 * w2)

            matrix_J = torch.zeros((b * c1 * c1, h2 * w2, h1 * w1), device=matrix.device)
            matrix_J[arange_repeated, idx_col] = matrix
            matrix_J = (
                matrix_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
                .movedim(-2, -3)
                .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
            )

            return matrix_J
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if val1 is None or val2 is None:
            raise NotImplementedError
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, m) for m in matrixes)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, m) for m in matrixes)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                return tuple(self._jacobian_wrt_input_sandwich_full_to_full(x1, val1, tmp) for tmp in tmps)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
            elif not diag_inp and diag_out:
                # full -> diag
                raise NotImplementedError
            elif diag_inp and diag_out:
                # diag -> diag
                return tuple(self._jacobian_wrt_input_sandwich_diag_to_diag(x1, val1, tmp) for tmp in tmps)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        assert c1 == c2

        tmp = tmp.reshape(b, c1, h2 * w2, c1, h2 * w2).movedim(-2, -3).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        Jt_tmp_J = torch.zeros((b * c1 * c1, h1 * w1, h1 * w1), device=tmp.device)
        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1), h2 * w2 * h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c1 * c1, h2 * w2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
        idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        idx_row = (
            idx.unsqueeze(2).expand(-1, -1, c1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2).movedim(-1, -2)
        )

        Jt_tmp_J[arange_repeated, idx_row, idx_col] = tmp
        Jt_tmp_J = (
            Jt_tmp_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
            .movedim(-2, -3)
            .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
        )

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, diag_tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, diag_tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        new_tmp = torch.zeros_like(x)
        new_tmp = new_tmp.reshape(b * c1, h1 * w1)

        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b * c2, h2 * w2)

        new_tmp[arange_repeated, idx] = diag_tmp.reshape(b * c2, h2 * w2)

        return new_tmp.reshape(b, c1 * h1 * w1)


class MaxPool3d(AbstractJacobian, nn.MaxPool3d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, d1 * h1 * w1, *jac_in_orig_shape[5:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * d2 * w2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :].reshape(*val.shape, *jac_in_orig_shape[5:])
        return jac_in


class Sigmoid(AbstractActivationJacobian, nn.Sigmoid):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = val * (1.0 - val)
        return jac


class ReLU(AbstractActivationJacobian, nn.ReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (val > 0.0).type(val.dtype)
        return jac

    def get_jacobian(self, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val)
            return torch.einsum("bi->bii", diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jvp(self, vector: Tensor, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val)
            return torch.einsum("bj,bj->bj", diag_jacobian, vector)
        elif wrt == "weight":
            return None

    def _vjp(self, vector: Tensor, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val)
            return torch.einsum("bi,bi->bi", vector, diag_jacobian)
        elif wrt == "weight":
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                diag_jacobian = (val > 0.0).type(val.dtype)
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, tmp, diag_jacobian)
            elif diag_inp and not diag_out:
                # diag -> full
                diag_jacobian = (val > 0.0).type(val.dtype)
                raise NotImplementedError
                # return torch.einsum("bi,bi,bi->bii", diag_jacobian, tmp, diag_jacobian)
            elif not diag_inp and diag_out:
                # full -> diag
                diag_jacobian = (val > 0.0).type(val.dtype)
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, tmp, diag_jacobian)
            elif diag_inp and diag_out:
                # diag -> diag
                diag_jacobian = (val > 0.0).type(val.dtype)
                return torch.einsum("bi,bi,bi->bi", diag_jacobian, tmp, diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            tmp11, tmp12, tmp22 = tmps
            if not diag_inp and not diag_out:
                # full -> full
                diag_jacobian_1 = (val1 > 0.0).type(val1.dtype)
                diag_jacobian_2 = (val2 > 0.0).type(val2.dtype)

                tmp11 = torch.einsum("bi,bik,bk->bik", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bik,bk->bik", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bik,bk->bik", diag_jacobian_2, tmp22, diag_jacobian_2)
            elif diag_inp and not diag_out:
                # diag -> full
                diag_jacobian_1 = (val1 > 0.0).type(val1.dtype)
                diag_jacobian_2 = (val2 > 0.0).type(val2.dtype)

                tmp11 = torch.einsum("bi,bi,bi->bii", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bi,bi->bii", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bi,bi->bii", diag_jacobian_2, tmp22, diag_jacobian_2)
            elif not diag_inp and diag_out:
                # full -> diag
                diag_jacobian_1 = (val1 > 0.0).type(val1.dtype)
                diag_jacobian_2 = (val2 > 0.0).type(val2.dtype)

                tmp11 = torch.einsum("bi,bii,bi->bi", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bii,bi->bi", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bii,bi->bi", diag_jacobian_2, tmp22, diag_jacobian_2)
            elif diag_inp and diag_out:
                # diag -> diag
                diag_jacobian_1 = (val1 > 0.0).type(val1.dtype)
                diag_jacobian_2 = (val2 > 0.0).type(val2.dtype)

                tmp11 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bi,bi->bi", diag_jacobian_2, tmp22, diag_jacobian_2)

            return (tmp11, tmp12, tmp22)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)


class PReLU(AbstractActivationJacobian, nn.PReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (val >= 0.0).type(val.dtype) + (val < 0.0).type(val.dtype) * self.weight.reshape(
            (1, self.num_parameters) + (1,) * (val.ndim - 2)
        )
        return jac


class ELU(AbstractActivationJacobian, nn.ELU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.ones_like(val)
        jac[x <= 0.0] = val[x <= 0.0] + self.alpha
        return jac


class Hardshrink(AbstractActivationJacobian, nn.Hardshrink):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.ones_like(val)
        jac[torch.logical_and(-self.lambd < x, x < self.lambd)] = 0.0
        return jac


class Hardtanh(AbstractActivationJacobian, nn.Hardtanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.zeros_like(val)
        jac[val.abs() < 1.0] = 1.0
        return jac


class LeakyReLU(AbstractActivationJacobian, nn.LeakyReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.ones_like(val)
        jac[x < 0.0] = self.negative_slope
        return jac


class Softplus(AbstractActivationJacobian, nn.Softplus):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.sigmoid(self.beta * x)
        return jac


class Tanh(AbstractActivationJacobian, nn.Tanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 1.0 - val**2
        return jac

    def get_jacobian(self, x: Tensor, val: Tensor, wrt="input"):
        if wrt == "input":
            diag_jacobian = self._jacobian(x, val)
            return diag_jacobian
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        b = x.shape[0]
        if wrt == "input":
            diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
            return torch.einsum("bj,bj->bj", diag_jacobian, vector)
        elif wrt == "weight":
            return None

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        b = x.shape[0]
        if wrt == "input":
            diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
            return torch.einsum("bi,bi->bi", vector, diag_jacobian)
        elif wrt == "weight":
            return None

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        b = x.shape[0]
        if wrt == "input":
            diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
            return torch.einsum("bi,bij->bij", diag_jacobian, matrix)
        elif wrt == "weight":
            return None

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        b = x.shape[0]
        if wrt == "input":
            diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
            return torch.einsum("bij,bj->bij", matrix, diag_jacobian)
        elif wrt == "weight":
            return None

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                diag_jacobian = (torch.ones(val.shape, device=val.device) - val**2).reshape(b, -1)
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
            elif from_diag and to_diag:
                # diag -> diag
                diag_jacobian_square = ((torch.ones(val.shape, device=val.device) - val**2) ** 2).reshape(
                    b, -1
                )
                return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        b = x1.shape[0]
        if val1 is None:
            val1 = self.forward(x1)
        if val2 is None:
            val2 = self.forward(x2)
        assert val1.shape == val2.shape
        if matrixes is None:
            matrixes = tuple(torch.ones_like(val1).reshape(b, -1) for _ in range(3))
            from_diag = True

        if wrt == "input":
            m11, m12, m22 = matrixes
            jac_1_diag = (torch.ones_like(val1) - val1**2).reshape(b, -1)
            jac_2_diag = (torch.ones_like(val2) - val2**2).reshape(b, -1)

            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    torch.einsum("bi,bij,bj->bij", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(
                    torch.einsum("bi,bii,bi->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("bi,bi,bi->bi", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [
                        (jac_1_diag, m11, jac_1_diag),
                        (jac_1_diag, m12, jac_2_diag),
                        (jac_2_diag, m22, jac_2_diag),
                    ]
                )
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, wrt="input", diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if wrt == "input":
            if not diag_inp and not diag_out:
                # full -> full
                diag_jacobian = torch.ones(val.shape, device=val.device) - val**2
                return torch.einsum("bi,bik,bk->bik", diag_jacobian, tmp, diag_jacobian)
            elif diag_inp and not diag_out:
                # diag -> full
                raise NotImplementedError
                # diag_jacobian = torch.ones(val.shape, device=val.device) - val ** 2
                # return torch.einsum("bi->bii", torch.einsum("bi,bi,bi->bi", diag_jacobian, tmp, diag_jacobian) )
            elif not diag_inp and diag_out:
                # full -> diag
                diag_jacobian = torch.ones(val.shape, device=val.device) - val**2
                return torch.einsum("bi,bii,bi->bi", diag_jacobian, tmp, diag_jacobian)
            elif diag_inp and diag_out:
                # diag -> diag
                diag_jacobian = torch.ones(val.shape, device=val.device) - val**2
                return torch.einsum("bi,bi,bi->bi", diag_jacobian, tmp, diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    def _jacobian_sandwich_multipoint(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        tmps: Tuple[Tensor, Tensor, Tensor],
        wrt="input",
        diag_inp: bool = True,
        diag_out: bool = True,
    ):
        if wrt == "input":
            tmp11, tmp12, tmp22 = tmps
            if not diag_inp and not diag_out:
                # full -> full
                diag_jacobian_1 = torch.ones(val1.shape, device=val1.device) - val1**2
                diag_jacobian_2 = torch.ones(val2.shape, device=val2.device) - val2**2

                tmp11 = torch.einsum("bi,bik,bk->bik", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bik,bk->bik", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bik,bk->bik", diag_jacobian_2, tmp22, diag_jacobian_2)

            elif diag_inp and not diag_out:
                # diag -> full
                diag_jacobian_1 = torch.ones(val1.shape, device=val1.device) - val1**2
                diag_jacobian_2 = torch.ones(val2.shape, device=val2.device) - val2**2

                tmp11 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bi,bi->bi", diag_jacobian_2, tmp22, diag_jacobian_2)
                raise NotImplementedError
                # return [torch.einsum("bi->bii", tmp) for tmp in [tmp11, tmp12, tmp22]]
            elif not diag_inp and diag_out:
                # full -> diag
                diag_jacobian_1 = torch.ones(val1.shape, device=val1.device) - val1**2
                diag_jacobian_2 = torch.ones(val2.shape, device=val2.device) - val2**2

                tmp11 = torch.einsum("bi,bii,bi->bi", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bii,bi->bi", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bii,bi->bi", diag_jacobian_2, tmp22, diag_jacobian_2)

            elif diag_inp and diag_out:
                # diag -> diag
                diag_jacobian_1 = torch.ones(val1.shape, device=val1.device) - val1**2
                diag_jacobian_2 = torch.ones(val2.shape, device=val2.device) - val2**2

                tmp11 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp11, diag_jacobian_1)
                tmp12 = torch.einsum("bi,bi,bi->bi", diag_jacobian_1, tmp12, diag_jacobian_2)
                tmp22 = torch.einsum("bi,bi,bi->bi", diag_jacobian_2, tmp22, diag_jacobian_2)
            return (tmp11, tmp12, tmp22)

        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return (None, None, None)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjk->bmk", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjm->bm", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.diag_embed(self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp_diag))

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = jac.view(x.shape[0], -1)
        tmp = jac**2 * tmp_diag
        return tmp


class ArcTanh(AbstractActivationJacobian, nn.Tanh):
    def forward(self, x: Tensor) -> Tensor:
        xc = x.clamp(
            -(1 - 1e-4), 1 - 1e-4
        )  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -1.0 / (x**2 - 1.0)
        return jac


class Reciprocal(AbstractActivationJacobian, nn.Module):
    def __init__(self, b: float = 0.0):
        super().__init__()
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        val = 1.0 / (x + self.b)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -((val) ** 2)
        return jac


class OneMinusX(AbstractActivationJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = 1 - x
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -torch.ones_like(x)
        return jac


class Sqrt(AbstractActivationJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = torch.sqrt(x)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 0.5 / val
        return jac


class RBF(nn.Module):
    def __init__(self, dim, num_points, points=None, beta=1.0):
        super().__init__()
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = nn.Parameter(points, requires_grad=False)
        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        else:
            self.beta = beta

    def __dist2__(self, x):
        x_norm = (x**2).sum(1).view(-1, 1)
        points_norm = (self.points**2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)  # NxM
        # if x.dim() is 2:
        #    x = x.unsqueeze(0) # BxNxD
        # x_norm = (x**2).sum(-1, keepdim=True) # BxNx1
        # points_norm = (self.points**2).sum(-1, keepdim=True).view(1, 1, -1) # 1x1xM
        # d2 = x_norm + points_norm - 2.0 * torch.bmm(x, self.points.t().unsqueeze(0).expand(x.shape[0], -1, -1))
        # return d2.clamp(min=0.0) # BxNxM

    def forward(self, x, jacobian=False):
        D2 = self.__dist2__(x)  # (batch)-by-|x|-by-|points|
        val = torch.exp(-self.beta * D2)  # (batch)-by-|x|-by-|points|

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        T1 = -2.0 * self.beta * val  # BxNxM
        T2 = x.unsqueeze(1) - self.points.unsqueeze(0)
        J = T1.unsqueeze(-1) * T2
        return J

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        if wrt == "input":
            jacobian = self._jacobian(x, val)
            return torch.einsum("bij,bjk->bik", jacobian, matrix)
        elif wrt == "weight":
            return None
