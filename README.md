# Apply WGAN-GP model
Image Size=64X64
 
# Init ENV
unzip img_celeb_align in ./data./img_celeb

cd images
mkdir images mdl_state report results

# To run 
# Defalut parameter(b1=0.5, b2=0.999, latent_dim=100, lr=0.0002, n_critic=5)
python  hw2_wgan_gp.py
