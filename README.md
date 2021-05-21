# MACC-FLOPS
Complexity of deep learning model using MACC and FLOPS


Deep Learning model is so complex in terms of Performance, Memory cost and Calculations(FLOPS).
When we consider any CNN network we should consider all these parameters.

**Convolutional layer MACC**
The input and output to convolutional layers are not vectors but three-dimensional feature maps of size H × W × C where H is the height of the feature map, W the width, and C the number of channels at each location.

Most convolutional layers used today have square kernels. For a conv layer with kernel size K, the number of MACCs is:

K × K × Cin × Hout × Wout × Cout
Here’s where that formula comes from:

for each pixel in the output feature map of size Hout × Wout,
take a dot product of the weights and a K × K window of input values
we do this across all input channels, Cin
and because the layer has Cout different convolution kernels, we repeat this Cout times to create all the output channels.

**Example: for a 3×3 convolution with 128 filters, on a 112×112 input feature map with 64 channels, we perform this many MACCs:

3 × 3 × 64 × 112 × 112 × 128 = 924,844,032
That’s almost 1 billion multiply-accumulate operations! got to keep that GPU busy…**

**Calculation of FLOPs of Convolutional Layer**
The unit often used in deep learning papers is GFLOPs, 1 GFLOPs = 10^9 FLOPs, that is: 1 billion floating point operations (1 billion, 000, 000, 000)
The floating point operations here are mainly W WRelated multiplications, and b bRelated additions, each W Wcorrespond W WMultiplication of the number of elements in each b bCorresponds to an addition, so it seems that the number of FLOPs and parameters are the same. But in fact, there is one place we overlooked, that is, the value on each layer of feature map is the result of processing through the same filter (weight sharing), which is an important feature of CNN (greatly reducing the amount of parameters) . So when we calculate FLOPs, we only need to multiply the size of the feature map on the basis of the parameters, that is, for a certain convolutional layer, the number of FLOPs is: [(K_h * K_w )* C_{in} + 1]*[(H_{out}*W_{out})* C_{out} ] = [(K_h * K_w * C_{in}) * C_{out} + C_{out}]*[H_{out}*W_{out}]= num_{parameter}*size_{output feature map} [(Kh​∗Kw​)∗Cin​+1]∗[(Hout​∗Wout​)∗Cout​]=[(Kh​∗Kw​∗Cin​)∗Cout​+Cout​]∗[Hout​∗Wout​]=numparameter​∗sizeoutputfeaturemap​,among them num_{parameter} numparameter​Indicates the number of parameters of this layer, size_{output feature map} sizeoutputfeaturemap​Indicates the two-dimensional size of the output feature map.

**github links for macc and flops**
https://github.com/sovrasov/flops-counter.pytorch
https://github.com/Lyken17/pytorch-OpCounter
