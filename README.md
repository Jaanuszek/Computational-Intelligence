# Project Ideas

## Image denoiser

### Why I chosed this topic? (Problem statement & motivation)

My master's thesis is about creating a graphics engine using Ray Tracing and Rasiosity algorithms. Path tracing is a ray tracing technique thas uses Monte Carlo method for calculating (approximating) the result (pixel color). Monte Carlo is stochastic algorithm and because of that, it creates a noisy results and therefore noisy images. The more iteration, the more accurate the result is, so in order to render a clear image we need to cast more rays, which is computationally expensive. Denoising resulted image might be a more optimal approach, which can save some valuable resources.

### Key terms, models, methods, and algorithms (small examples/diagrams)

* **MOEDELS** </br>
    - Deep CNNs
    - Autoencoders - When tou have both original and noisy image
    - GANs
    - Noise2Noise - Network can be trained using pairs of noisy images
    - Noise2Void - Only requires noisy images. Signal has a structure, noise does not. We can't predict noise by looking at surroiunding pixels, but we can predict structures.
    - Unet (?)

### baseline plan (data, model(s), metrics)

I want to use different methods to check what approach gives the best result (visual effect, accuracy loss, resolution loss, visable artifacts). I'll try to teach model using semi-supervised and unsupervised(?) learning approaches. Will use [Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) dataset, noisy images will be crated adding some artificial noise.

* Metrics:
    - PSNR (Peak Signal to Noise Ratio)
    - MSE (Mean Squared Error)
    - SSIM (Structural Similarity Index)


### Constraints & resources, risks, and a rough timeline

### References

### Links to candidate datasets

## Optimizing model using QAT

### Why I chosed this topic? (Problem statement & motivation)

### Key terms, models, methods, and algorithms (small examples/diagrams)

### baseline plan (data, model(s), metrics)

### Constraints & resources, risks, and a rough timeline

### References

### Links to candidate datasets