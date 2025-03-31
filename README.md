# DTV-NET: An Iterative Reconstruction Network for Incomplete Projections in Static CT

Official PyTorch Implementation for:

DTV-NET: An Iterative Reconstruction Network for Incomplete Projections in Static CT


## Abstract
The Nanovision static CT, a novel computed tomography (CT) scanning system, employs a multi-source array and a flat-panel detector array fixed on two parallel planes with a constant offset.  Unlike conventional CT systems, this static CT acquires full projection views in axial scanning mode using a focus-shifting technique combined with small-angle gantry rotation.  This unique scanning protocol limits the angular range of each source, enabling complete scan acquisition. However, the large cone angle between the sources and the detector, combined with the uneven clustering of projections inherent in multi-source acquisition, leads to significantly incomplete projections. Consequently, significant cone-beam artifacts and uneven sparse-angle artifacts coexist, degrading the reconstructed image quality. To address these issues, this paper proposes a deep iterative network based on directional Total Variation regularization (DTV-Net). DTV-Net incorporates DTV as a regularization term within the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) framework, achieving both artifact suppression and rapid convergence. Specifically, it employs an encoder-decoder architecture and a Head Attention Block (HAB) module to adaptively adjust threshold parameters in the gradient space, effectively removing redundant gradient information corresponding to artifacts. During end-to-end training, we integrated the ASTRA toolbox with tensorized representations and introduced a Tensorized Projection Operator (TPO) tailored for the multi-flat-panel detector array, optimizing iterative forward and backward projections. Extensive experiments demonstrate that the proposed DTV-Net algorithm outperforms prior art solutions on both simulation and clinical data. 

## Model and part of data
You can download from Baidu disk: https://pan.baidu.com/s/18s36ORi3Kc3wgyFIL9TUxQ?pwd=3w3m