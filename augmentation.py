import Augmentor


def augment_data(base_path, batch_size=10, img_size=300, isval=False, isgen=False):
    """
    Generates augmented images from a class-based directory structure
    :param base_path: path to images
    :param batch_size:batch size for generator
    :param img_size: target image size
    :param isval: True if validation data
    :param isgen: True if generator required
    :return: a generator or None
    .. note:: expected file structure inside base path
    -
        |-Rook/
        |-Knight/
        |-Bishop/
        |-King/
        |-Pawn/
        |-Queen/
    """
    p = Augmentor.Pipeline(base_path)
    if not isval:
        p.rotate(probability=0.8, max_left_rotation=20, max_right_rotation=20)
        p.flip_left_right(probability=0.8)
        p.zoom_random(probability=0.8, percentage_area=0.9)
        p.flip_top_bottom(probability=0.3)
        p.crop_random(0.5, 0.9)
        p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    p.greyscale(probability=1.0)
    p.resize(probability=1.0, width=img_size, height=img_size)
    if isgen:
        gen = p.keras_generator(batch_size=batch_size, scaled=True)
        return gen, len(p.augmentor_images)
    else:
        print(len(p.augmentor_images)*3)
        p.sample(len(p.augmentor_images)*3)


if __name__ == '__main__':
    augment_data('data/train', img_size=300)
