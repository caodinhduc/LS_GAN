This repo implements LS GAN with training dataset is LSUN dataset(church outdoor)

it contains:
1. data_loader.py supports pytorch data loader
2. model.py defines generator and discriminator
3. train.py
4. utils.py provides necessary libraries

Finally, in inference.py, you are able to run generator to create fake images, please check carefully several paths.
I provided temporary checkpoint in folder "checkpoint/"

Model architectures:
<div align="center">
  <img src="images/generator.png" width="800px" />
  <p>Generator</p>
</div>

<div align="center">
  <img src="images/discriminator.png" width="800px" />
  <p>Discriminator</p>
</div>

Several generations:

<div align="center">
  <img src="images/fake1.png" width="400px" />
  <p>Fake images example</p>
</div>

<div align="center">
  <img src="images/fake2.png" width="400px" />
  <p>Fake images example</p>
</div>