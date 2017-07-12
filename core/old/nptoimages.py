from PIL import Image
import numpy as np
# sourcefil = sys.argv[1]
images = np.load('trainRandomTRs.npy')
for i in range(images.shape[0]):
    img = (((images[i] - images[i].min()) / (images[i].max() - images[i].min())) * 255.9).astype(np.uint8)
    img=Image.fromarray(img, 'L')
    na=str(i)
    while len(na)<6:
    	na='0' +na
    img.save('images/image' + na +'.jpg')
