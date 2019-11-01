from project.UNet import UNet

def segment(img):
    """
    Semantically segment an image
    img: an uint8 numpy of size (w,h,3)
    return: a numpy integer array of size (w,h), where the each entry represent the class id
    please refer to data/color_map.json for the id <-> class mapping
    """
    net = UNet(3, 1).cuda()
    segment = net.forward(img.cuda())

    return segment