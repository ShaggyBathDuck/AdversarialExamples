import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from generator import fgs
from torchvision import models
import os, os.path
from generator import util

netImageLabels = {
    'cat': 282,  # tiger cat
    'dog': 245,  # French bulldog
    'orange': 950,
    'banana': 954,
    'hotdog': 934,
    'icecream': 928
}

model = models.resnet34(pretrained=True)

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')
IMAGES_OUT = os.path.join(os.path.dirname(__file__), 'images_out')
ADV_SUFFIX = '_adv'

lock = threading.RLock()


def generate_adversarial_image(dir_path, filename, target_label, label):
    print(filename)

    image_path = os.path.join(dir_path, filename)
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.ANTIALIAS)

    tensor = util.preprocess(img)

    with lock:
        adversarial_tensor = fgs.fgs(model, tensor, netImageLabels[target_label],
                                     targeted=True, alpha=0.01, iterations=10,
                                     use_cuda=False)

    adversarial_image = util.postprocess(adversarial_tensor)

    org_dir = os.path.join(IMAGES_OUT, label)
    adv_dir = os.path.join(IMAGES_OUT, label + ADV_SUFFIX)

    if not os.path.exists(org_dir):
        os.makedirs(org_dir)

    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)

    adversarial_image.save(os.path.join(adv_dir, filename))
    img.save(os.path.join(org_dir, filename))
    return


if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=4)

    for group in os.listdir(IMAGES_PATH):
        group_dir = os.path.join(IMAGES_PATH, group)
        labels = os.listdir(group_dir)

        label_dir = [os.path.join(group_dir, labels[0]), os.path.join(group_dir, labels[1])]

        for image in os.listdir(label_dir[0]):
            executor.submit(generate_adversarial_image, label_dir[0], image, target_label=labels[1], label=labels[0])

        for image in os.listdir(label_dir[1]):
            executor.submit(generate_adversarial_image, label_dir[1], image, target_label=labels[0], label=labels[1])

