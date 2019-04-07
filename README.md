# RSforWordTranslation
## Reconstructed Similarity for Faster GANs-Based Word Translation to Mitigate Hubness
In machine word translation, nearest neighbor (NN) retrieval is able to search the best-Ktranslation candidates as linguistic labels of a source query from a unified multilingual semantic feature space. However, NN is polluted by hubs in the high-dimensional feature space. Many proposed approaches remove hubs in the list of translation candidates to relieve this problem. However, this approach to eliminating hubs is flawed because they also have corresponding translations. To address this issue, we propose a novel reconstructed similarity (RS) retrieval for the neural machine word translation model to mitigate the hubness problem regardless of whether it is a hub. Different from previous work, RS reduces the impact of hubness pollution in dense and high-dimensional space and allows the hubs to have the same probability as the target candidates without being inappropriately excluded. In addition, RS improves the quality of bilingual dictionaries by measuring the bilateral similarity of the bilingual and monolingual distance of each of the source query embeddings. Additionally, to model the unsupervised machine word translation, we introduce GANs to map the source and target word distribution into a shared semantic space. We also construct a tiny GAN topology for neural machine word translation, which is at least 52 faster than previous GAN-based models. To further align cross-lingual embedding distributions, we provide orthogonal Procrustes mapping, global-awareness of the transformation matrix and rescaling of the target embeddings as flexible and optional multirefinements. The results show that our model outperforms the state of the art by nearly 4% in distant languages such as English to Chinese and English to Finnish. 
Compared with a precision@1 of 47.00% from English to Finnish, our model obtains a precision@1 of 47.53\% and achieves state-of-the-art results in a fully unsupervised form. Moreover, our model achieves competitive results in the shortest time among GAN-based models, which easily trade off between speed and accuracy. 

**Our main contributions include the following:**
- We first propose the global-awareness of the transformation matrix W that fuses global and local weights, which obtains a subtle improvement.
- We introduce a novel reconstructed similarity retrieval for selecting the best-k target candidates that evaluates the likelihood of candidates combining bilingual and monolingual distances. This approach eliminates the hubness pollution without excluding the hubs. Additionally, it extends previous work and achieves competitive performances.
- Our model significantly reduces model training time, which only takes a few minutes with a single CPU and is approximately 52X~105X faster than previous GAN-based models.		
- Our unsupervised word translation model obtains the state of the art on the bilingual lexicon extraction task. It reaches a precision@1 of 47.53%  translating English to Finnish and outperforms supervised methods without a single parallel sign.



 表格      | 第一列     | 第二列     
 -------- | :-----------:  | :-----------: 
 第一行     | 第一列     | 第二列    

## Visualization
we visualize three translation scenarios by projecting mapped bilingual word embeddings from 300 dimensions to 2D at prediction time. 
This visualization shows the process of how RS works from an experimental view and fully demonstrates that RS promotes the confidence of more appropriate target translations without considering whether it is a hub. The word embeddings are projected from 300-dim to the 2D plane using PCA. The five vectors in Spanish (black) are the 5 nearest neighbors of the source query, and the values below the red word labels are the distance from the query. Another five vectors in English (red) are the 5 nearest neighbors of the target candidate that is in the best-k list, and the values below the black labels are the distance from the candidate. In addition, the table in the top right corner reports the similarity of the query and the back-translation value is shown with red labels.
![](https://github.com/djzgroup/RSforWordTranslation/blob/master/visualizaition.jpg)

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
