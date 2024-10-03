#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# modification by VITA, Kevin 
# added the f_count, a count flag to count the number of time a guassian is activated.  



def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_densify,
    sh,
    colors_precomp,
    normals_precomp,
    semantics_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    dirs,
    inside,
    raster_settings,
):
    if raster_settings.f_count == 0:
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            means2D_densify,
            sh,
            colors_precomp,
            normals_precomp,
            semantics_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            dirs,
            inside,
            raster_settings,
        )
    elif raster_settings.f_count == 1:
        return _RasterizeGaussians.forward_count(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        )
    elif raster_settings.f_count == 2:
        return _RasterizeGaussians.forward_visi(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        )
    elif raster_settings.f_count == 3:
        return _RasterizeGaussians.forward_visi_acc(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_densify,
        sh,
        colors_precomp,
        normals_precomp,
        semantics_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        dirs,
        inside,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            # normals_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )
        gaussians_count, important_score, num_rendered, outmap, radii, geomBuffer, binningBuffer, imgBuffer = None, None, None, None, None, None, None, None
        # Invoke C++/CUDA rasterizer
        # TODO(Kevin): pass the count in, but the output include a count list 
        if raster_settings.f_count == 1:
            args = args + (raster_settings.f_count,)
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    
                    gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
            
        elif raster_settings.f_count == 2:
            args = args + (raster_settings.f_count,)
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    
                    gaussians_count, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                gaussians_count, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians(*args)
            
        elif raster_settings.f_count == 3:
            args = args + (raster_settings.f_count,)
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    
                    gaussians_count, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
        
        else:
            args = (
                raster_settings.bg,
                means3D,
                colors_precomp,
                normals_precomp,
                semantics_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                dirs,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                inside,
                raster_settings.debug
            )
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    num_rendered, outmap, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                num_rendered, outmap, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        depth, alpha = outmap[3:4], outmap[7:8]
        ctx.save_for_backward(colors_precomp, normals_precomp, semantics_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, depth, dirs, inside)
        
        if raster_settings.f_count == 1:
            return gaussians_count, important_score, color, radii 
        elif raster_settings.f_count == 2:
            return gaussians_count, important_score, color, radii
        return outmap, radii

    @staticmethod
    def forward_count(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        assert(raster_settings.f_count == 1)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.f_count
        )
        # gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = None, None, None, None, None, None, None, None
        # Invoke C++/CUDA rasterizer
        # TODO(Kevin): pass the count in, but the output include a count list 
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
                   
        return gaussians_count, important_score, color, radii 

    @staticmethod
    def forward_visi(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        assert(raster_settings.f_count == 2)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.f_count
        )
        # gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = None, None, None, None, None, None, None, None
        # Invoke C++/CUDA rasterizer
        # TODO(Kevin): pass the count in, but the output include a count list 
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                gaussians_count, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians(*args)
            # gaussians_count, _, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
        
        return gaussians_count, important_score, color, radii 

    @staticmethod
    def forward_visi_acc(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        assert(raster_settings.f_count == 3)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.f_count
        )
        # gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = None, None, None, None, None, None, None, None
        # Invoke C++/CUDA rasterizer
        # TODO(Kevin): pass the count in, but the output include a count list 
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
            # gaussians_count, _, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
        
        return gaussians_count, radii 

    @staticmethod
    # def backward(ctx, grad_color, grad_radii, grad_depth, grad_depth_var, grad_median_depth, grad_normal, grad_semantics, grad_alpha):
    # def backward(ctx, grad_color, grad_radii, grad_depth, grad_normal, grad_semantics, grad_alpha):
    def backward(ctx, grad_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, normals_precomp, semantics_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, depth, dirs, inside = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                normals_precomp,
                semantics_precomp,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                dirs,
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_color,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                depth,
                inside,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_densify, grad_colors_precomp, grad_normals_precomp, grad_semantics_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_means2D_densify, grad_colors_precomp, grad_normals_precomp, grad_semantics_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_densify,
            grad_sh,
            grad_colors_precomp,
            grad_normals_precomp,
            grad_semantics_precomp, # .unsqueeze(1)
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    f_count : int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_densify, opacities, shs = None, colors_precomp = None, normals_precomp=None, semantics_precomp = None, scales = None, rotations = None, cov3D_precomp = None, dirs=None, inside=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        
        if normals_precomp is None:
            normals_precomp = torch.Tensor([])

        if semantics_precomp is None:
            semantics_precomp = torch.Tensor([])
            
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if dirs is None:
            dirs = torch.Tensor([])
        if inside is None:
            inside = torch.ones(means3D.shape[0], device=means3D.device, dtype=torch.bool)

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_densify,
            shs,
            colors_precomp,
            normals_precomp,
            semantics_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            dirs,
            inside,
            raster_settings,
        )
        
    #TODO(Kevin add counter version of forward)
    def forward_count(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        ) 

    #TODO(Kevin add counter version of forward)
    def forward_visi(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        ) 

    #TODO(Kevin add counter version of forward)
    def forward_visi_acc(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        ) 
