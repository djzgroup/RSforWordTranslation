# RSforWordTranslation
Reconstructed Similarity for Faster GANs-Based Word Translation to Mitigate Hubness

**Our main contributions include the following:**
- We first propose the global-awareness of the transformation matrix $W$ that fuses global and local weights, which obtains a subtle improvement.
- We introduce a novel reconstructed similarity retrieval for selecting the best-$k$ target candidates that evaluates the likelihood of candidates combining bilingual and monolingual distances. This approach eliminates the hubness pollution without excluding the hubs. Additionally, it extends previous work and achieves competitive performances.
- Our model significantly reduces model training time, which only takes a few minutes with a single CPU and is approximately 52$\times$$\sim$105$\times$ faster than previous GAN-based models.		
- Our unsupervised word translation model obtains the state of the art on the bilingual lexicon extraction task. It reaches a precision@1 of 47.53\%  translating English to Finnish and outperforms supervised methods without a single parallel sign.

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
