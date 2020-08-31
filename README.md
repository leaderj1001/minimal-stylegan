# minimal-stylegan (A Style-Based Generator Architecture for Generative Adversarial Networks)
  - [Paper Link](https://arxiv.org/pdf/1812.04948.pdf)
  - Author: Tero Karras, Samuli Laine, Timo Aila
  - Organization: NVIDIA
  - [Reference Code URL](https://github.com/rosinality/style-based-gan-pytorch)
  
 ## Method
   1. prepare dataset (FFHG Dataset) <br>
     - Use 0-2999 photos (Total: 3000) <br>
   ```
   python prepare.py
   ```
   
   2. train
   ```
   python main.py
   ```
   
   3. eval
   ```
   python eval.py
   ```
   
 ## Results
   - Batch_size = 30
   - Loss = 'wgan-gp'
   - Training Time: 75 Hours, (80000 Iterations)
   - 1 GPU, image size: 128x128
   ![eval](https://user-images.githubusercontent.com/22078438/91691175-28cd6480-eba2-11ea-9e7f-89de36e80b5c.png)
