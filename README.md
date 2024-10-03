# Differential Gaussian Rasterization

**NOTE**: this is a modified version to support depth & alpha rendering (both forward and backward) from the [original repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization). 

```python
rendered_out, radii = rasterizer(
    means3D = means3D,
    means2D = means2D,
    means2D_densify = means2D_densify,
    shs = shs,
    colors_precomp = colors_precomp,
    normals_precomp = normals_precomp,
    semantics_precomp = sem_feats,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp,
    dirs = dirs,
    inside = inside
)
```


Used as the rasterization engine for the paper "VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{chen2024vcr,
        author    = {Chen, Hanlin and Wei, Fangyin and Li, Chen and Huang, Tianxin and Wang, Yunsong and Lee, Gim Hee},
        title     = {VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction},
        journal   = {arXiv preprint arXiv:2406.05774},
        year      = {2024},
}</code></pre>
  </div>
</section>
