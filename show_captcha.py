from create_captcha import create_captcha, np
from segment_image import segment_image

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'tk')
from matplotlib import pyplot as plt

image = create_captcha("VIDA", shear = 0.2)
plt.imshow(image, cmap = 'Greys')
