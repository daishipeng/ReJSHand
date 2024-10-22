# ReJSHand
The code of "A Lightweight Network Integrating Refined Joint and Skeleton Features for Real-Time Hand Pose Estimation and Mesh Reconstruction"

We refer to the code of [simpleHand](https://github.com/patienceFromZhou/simpleHand.git).  

![image](/configs/res.png)   

We propose ReJSHand, which surpasses the SoTA methods and is more computationally efficient.  

# Getting Started
## Training
Training requires the FreiHAND dataset. Please download it from [here](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and refer to the [FreiHAND toolbox](https://github.com/lmb-freiburg/freihand) to perform the MANO model and generate vertices.


During training, you can set the parameter n according to your device and use the following command to train:  
```
torchrun --nproc_per_node n train.py --resume
```

