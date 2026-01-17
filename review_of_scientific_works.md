# Project Ideas

## Image denoiser

### Why I chosed this topic? (Problem statement & motivation) TODO WHY THIS MATTERS??

My master's thesis is about creating a graphics engine using Ray Tracing and Radiosity algorithms. Path tracing is a ray tracing technique thas uses Monte Carlo method for calculating (approximating) the result (pixel color). Monte Carlo is stochastic algorithm and because of that, it creates a noisy results and therefore noisy images. The more iteration, the more accurate the result is, so in order to render a clear image we need to cast more rays, which is computationally expensive. Denoising resulted image with AI might be a more optimal approach, which can save some valuable resources.

### Key terms, models, methods, and algorithms (small examples/diagrams)

* **MOEDELS** </br>
    - CNN
    - Autoencoders - When you have both original and noisy image
    - Unet (?)
    - Noise2Noise - Network can be trained using pairs of noisy images
    - Noise2Void - Only requires noisy images. Signal has a structure, noise does not. We can't predict noise by looking at surroiunding pixels, but we can predict structures.

### Baseline plan (data, model(s), metrics)

I want to use different methods to check what approach gives the best result (visual effect, accuracy loss, resolution loss, visable artifacts). I'll try to teach model using semi-supervised and unsupervised(?) learning approaches. Will use [Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) dataset, noisy images will be crated adding some artificial noise.

* Metrics:
    - PSNR (Peak Signal to Noise Ratio)
    - MSE (Mean Squared Error)
    - SSIM (Structural Similarity Index) - Metric that quantifies image quality degradation caused by processing the data compression. It measures the difference between two images reference image - test image.


### Constraints & resources, risks, and a rough timeline

From the risk perspective, I am worried the most about time needed for teaching a model, so it might be important to convert images to grey scale in order to make learning faster and easier to test.

### References

* https://www.kaggle.com/code/ahmedelsayedtaha/denoising-images-using-autoencoder
* https://www.kaggle.com/code/davidesavarro/image-denoiser
* https://raver119.medium.com/denoising-images-with-deep-learning-9124d36fda0a
* https://github.com/Cydral/AI-Image-Denoiser
* https://blogs.nvidia.com/blog/what-is-denoising/
* https://github.com/SecretMG/UNet-for-Image-Denoising
* https://www.kaggle.com/code/andreipaulavets/byu-denoising-cryo-et-with-noise2void

### Links to candidate datasets

* https://www.kaggle.com/datasets/moltean/fruits/data
* https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
* https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset

## Optimizing model using QAT

### Why I chosed this topic? (Problem statement & motivation)

In the era of miniaturization, where small devices are faster than very expensive computers from a few years ago, there is a big demand packing the most functionalities in those small devices. Unfortunately, the most of good AI models, that use FP32 or BF32 precision take up a lot of space and require high computing power, which translates into higher energy/battery consumption. In mobile phones or battery-powered embedded devices, every second of battery life couts, which is why optimizing AI models through quantization is crucial. At this point, many AI functiosn on phones rely on sending queries to the cloud, so in order to use these functions, the device must have access to the internet. But what if we could optimize the model to such an etent that it would take up muc less space and be very similar in performance to the original non-quantized model? We could then afford to store and use this model directly in the device's memory, alowing AI fetures to be used without internet access and with lower battery consumption.

Why optimize model?

* Size reuction:
    - Smaller storage size
    - Smaller download size
    - Less memory usage - Models use less RAM when they are running, which frees up memory for other parts of yout application to use.
* Latency reduction
    - Quantization can be used to reduce latency by simplifying the calculations that occur during inference, potentially at the expense of some accuracy
* Accelerator compatibility
    - Some edge devices, e.g. edge TPU (Tensor Processing Unit) - might inference faster with models that have been correctly optimized - quantized in a specific way.

### Key terms, models, methods, and algorithms (small examples/diagrams)

* Quantization
    - QAT (Quantization-Aware Training) - Simulates low-precision inference-time computation in the forward pass of the training process. That means, we are changing the weights precission to INT8, but we still use FP32, which introduce quantization error as noise during the training, which optimizator in backward pass is trying to reduce using FP32 precission.
    - PTQ (Post-Training Quantization) - Is just quantizing already trained (float precission) model
        * Static Quantization - Requires a calibration step, uses fixed quantization parameters, offers faster inference with purely integer arithmetic, and is ideal for scenarios with known and stable input data distributions.
        * Dynamic range Quantization - Is the simplest form of PTQ that only quantizes the weights from floating point to integer. FP32 (4B) -> INT8 (1B) it is 4x smaller and therefore has speedup in CPU operations.
            - Weights are stored and computed in int8, but activations remain in floatin point until they are used in computations
            - Does not need a calibration step
* Weight Pruning - trims parameters within a model that has very less impact on the performance of the model. We can just cut the unnecessary weights that does not matter (sparse network - rzadsza sieć). It reduces the model sizes without a big accuracy loss.
* Fine-tuning - A process in which we take an already trained model (known as a base model or pretrained model) and train it further on a new, often smaller dataset, adapting it to a specific task.

### Baseline plan (data, model(s), metrics)

The idea is:

1. Create and train a model for image recognision in FP32
2. Create and train a model in the same way but using some quantization methods (Using QAT, PTQ, DRQ, or combination of them)
3. Compare metrics of those models
4. (OPTIONAL) Weight Pruning

The conclusion, I will include:
- Is the loss of model accuracy due to quantization acceptable?
- Is the model size reduction significant?
- Is model inference faster?

- Metrics:
    * Accuracy
    * Model size
    * Latency
    * Throughput
    * RAM / VRAM usage

### Constraints & resources, risks, and a rough timeline

* PyTorch
* TensorFlow,
* TensorFLow Lite - framework that converts a pre-trained model in TensorFlow to a special format that can be optimized for speeed or storage

### References

* https://www.kaggle.com/code/ashusma/understanding-tf-lite-and-model-optimization
* https://selek.tech/posts/static-vs-dynamic-quantization-in-machine-learning/

### Links to candidate datasets

* https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
* https://www.kaggle.com/datasets/alessiocorrado99/animals10