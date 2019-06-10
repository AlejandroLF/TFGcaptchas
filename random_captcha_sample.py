from create_captcha import create_captcha, np
from segment_image import segment_image

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'tk')
from matplotlib import pyplot as plt

from sklearn.utils import check_random_state
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)

    return create_captcha(letter, shear=shear, size=(20,20)), letters.index(letter)


image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
print("The target for this image is: {0} ({1})".format(target, letters[target]))
