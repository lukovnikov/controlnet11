import math
import torch
from torch.nn.functional import conv2d

from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import interpolate
from perlin_noise import perlin_noise


def hardblur(pilimg, radius=5, colors=None, bgr_color=None):    # if colors is None, then we do all of them 
    # img = to_tensor(pilimg)
    img = pilimg
    intimg = (img * 255).to(torch.long)
    colorcodes = intimg[2] * 256*256 + intimg[1] * 256 + intimg[0]
    bgrcode = (bgr_color * 255).to(torch.long)
    bgrcode = bgrcode[0] + bgrcode[1] * 256 + bgrcode[2] * 256*256
    uniquecolorcodes = set(colorcodes.unique().cpu().numpy()) - {bgrcode.cpu().item()}
    if colors is None:
        colors = uniquecolorcodes
    
    # create uniform circular kernel
    kernel = torch.zeros(1, 1, radius * 2 + 1, radius * 2 + 1)
    for i in range(kernel.shape[2]):
        for j in range(kernel.shape[3]):
            x, y = i - radius, j - radius
            if math.sqrt(x**2 + y**2) <= radius:
                kernel[:, :, i, j] = 1
    
    # generate separate masks for every color
    colorimgs = []
    for colorcode in colors:
        colorimg = (colorcodes == colorcode).float()
        colorimg = (conv2d(colorimg[None], kernel, padding="same") > 0).float()
        colorimgs.append(colorimg)
        
    # merge masks into overlapping colors
    normalizer = sum(colorimgs)
    ret = torch.zeros_like(img)
    
    for colorcode, colorimg in zip(colors, colorimgs):
        rgb_tensor = torch.tensor([colorcode % 256, (colorcode // 256) % 256, colorcode // (256*256) ])
        ret += colorimg * rgb_tensor[:, None, None]
    ret /= normalizer
    
    ret = torch.where(normalizer > 0, ret, img)
    ret /= 255
    
    return ret


def generate_random_precision_map(shape=(512, 512), gridsize=2, rescale=1):
    _shape = tuple([x//rescale for x in shape])
    noise = perlin_noise(grid_shape=(gridsize, gridsize), out_shape=_shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = interpolate(noise[None, None], shape, mode="bilinear")[0, 0]
    return noise


def create_circular_kernel(radius=5):
    kernel = torch.zeros(1, 1, radius * 2 + 1, radius * 2 + 1)
    for i in range(kernel.shape[2]):
        for j in range(kernel.shape[3]):
            x, y = i - radius, j - radius
            if math.sqrt(x**2 + y**2) <= radius:
                kernel[..., i, j] = 1
    return kernel


def variable_hardblur(pilimg, 
                      rmap=32,
                      gridsize=2,
                      colors=None, 
                      bgr_color=torch.tensor([0, 0, 0]), 
                      device=torch.device("cpu"),
                      rescale=2,
                      smoothing=11,
                      rescale_rmap=8,
                    #   rescale_img=2,
                     ):
    img = pilimg.to(device)
    original_shape = img.shape[-2:]
    
    if isinstance(rmap, int):
        _rmap = generate_random_precision_map(gridsize=gridsize, shape=original_shape, rescale=rescale_rmap)
        rmap = (_rmap * rmap)
    rmap = rmap.to(device)
    
    inner_shape = tuple([x // rescale for x in original_shape])
    
    intimg = (img * 255).to(torch.long)
    colorcodes = intimg[2] * 256*256 + intimg[1] * 256 + intimg[0]
    bgrcode = (bgr_color * 255).to(torch.long)
    bgrcode = bgrcode[0] + bgrcode[1] * 256 + bgrcode[2] * 256*256
    uniquecolorcodes = set(colorcodes.unique().cpu().numpy()) - {bgrcode.cpu().item()}
    if colors is None:
        colors = uniquecolorcodes
        
    _rmap = (interpolate(rmap[None, None], inner_shape, mode="bilinear")[0, 0] / rescale).long()
    # _rmap = torch.clamp_min(_rmap, 1)
    
    # generate separate masks for every color
    regionmasks = []
    for colorcode in colors:
        current_region_mask = (colorcodes == colorcode).float()                                # (H x W)
        
        _current_region_mask = interpolate(current_region_mask[None, None], inner_shape, mode="nearest")[0, 0]
        kernelsizes = _rmap * _current_region_mask                                               # (H x W)
        for kernelsize in list(kernelsizes.unique().long().cpu().numpy()):
            if kernelsize < 1:
                continue
            kernelsize_mask = kernelsizes == kernelsize
            kernel = create_circular_kernel(kernelsize).to(device)
            _expanded_region_mask = (conv2d(kernelsize_mask[None].float(), kernel, padding="same") > 0)
            expanded_region_mask = interpolate(_expanded_region_mask[None].float(), original_shape, mode="bicubic")[0]
            if smoothing > 0:
                expanded_region_mask = gaussian_blur(expanded_region_mask, smoothing)
                expanded_region_mask = expanded_region_mask > 0.5
            expanded_region_mask = expanded_region_mask.bool()
            current_region_mask = torch.maximum(current_region_mask, expanded_region_mask)
        regionmasks.append(current_region_mask)
        
    # merge masks into overlapping colors
    normalizer = sum(regionmasks)
    ret = torch.zeros_like(img).to(device)
    
    for colorcode, colorimg in zip(colors, regionmasks):
        rgb_tensor = torch.tensor([colorcode % 256, (colorcode // 256) % 256, colorcode // (256*256) ]).to(device)
        ret += colorimg * rgb_tensor[:, None, None]
    ret /= normalizer
    
    ret = torch.where(normalizer > 0, ret, img)
    ret /= 255
    
    return ret


def variable_softblur(pilimg, 
                      rmap=32,
                      gridsize=2,
                      colors=None, 
                      bgr_color=torch.tensor([0, 0, 0]), 
                      device=torch.device("cpu"),
                      rescale=2,
                    #   smoothing=11,
                      rescale_rmap=8,
                    #   rescale_img=2,
                    ):
    img = pilimg.to(device)
    original_shape = img.shape[-2:]
    
    if isinstance(rmap, int):
        _rmap = generate_random_precision_map(gridsize=gridsize, shape=original_shape, rescale=rescale_rmap)
        rmap = (_rmap * rmap)
    rmap = rmap.to(device)
    
    inner_shape = tuple([x // rescale for x in original_shape])
    
    intimg = (img * 255).to(torch.long)
    colorcodes = intimg[2] * 256*256 + intimg[1] * 256 + intimg[0]
    bgrcode = (bgr_color * 255).to(torch.long)
    bgrcode = bgrcode[0] + bgrcode[1] * 256 + bgrcode[2] * 256*256
    uniquecolorcodes = set(colorcodes.unique().cpu().numpy()) - {bgrcode.cpu().item()}
    if colors is None:
        colors = uniquecolorcodes
        
    _rmap = (interpolate(rmap[None, None], inner_shape, mode="bilinear")[0, 0] / rescale).long()
    _rmap = torch.clamp_min(_rmap, 1)
    
    # generate separate masks for every color
    regionmasks = []
    for colorcode in colors:
        current_region_mask = (colorcodes == colorcode).float()                                # (H x W)
        current_region_mask_out = torch.zeros_like(current_region_mask)
        _current_region_mask = interpolate(current_region_mask[None, None], inner_shape, mode="nearest")[0, 0]
        kernelsizes = _rmap * _current_region_mask                                               # (H x W)
        for kernelsize in list(kernelsizes.unique().long().cpu().numpy()):
            if kernelsize < 1:
                continue
            kernelsize_mask = kernelsizes == kernelsize
            kernel = create_circular_kernel(kernelsize).to(device)
            # if kernelsize < 1:
            kernel = kernel / (kernel.sum() + 1e-6)
            _expanded_region_mask = conv2d(kernelsize_mask[None].float(), kernel, padding="same")
            expanded_region_mask = interpolate(_expanded_region_mask[None].float(), original_shape, mode="nearest")[0]
            current_region_mask_out = current_region_mask_out + expanded_region_mask

        regionmasks.append(current_region_mask_out)
        
    # merge masks into overlapping colors
    normalizer = sum(regionmasks)
    ret = torch.zeros_like(img).to(device)
    
    for colorcode, colorimg in zip(colors, regionmasks):
        rgb_tensor = torch.tensor([colorcode % 256, (colorcode // 256) % 256, colorcode // (256*256) ]).to(device)
        ret += colorimg * rgb_tensor[:, None, None]
    ret /= normalizer
    
    ret = torch.where(normalizer > 0, ret, img)
    ret /= 255
    
    return ret