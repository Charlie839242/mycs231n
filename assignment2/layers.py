import numpy as np

"""
Detail of a Fully Connected Layer.

The input x has shape (N, d_1, ..., d_k) where N indicates the number of
examples and each example x[i] has shape (d_1, ..., d_k).
Therefore x is reshaped into a matrix with size (N, D) where
D = d_1 * ... * d_k.

Layer inputs:
- x: A numpy array of input data, of shape (N, d_1, ..., d_k) → X: (N, D)
- w: A numpy array of weights, of shape (D, M)
- b: A numpy array of biases, of shape (M,)

Layer output:
- out: output, of shape (N, M)

forward: 
- out = x × w + b

backward:
- db = np.sum(upstream grad, axis=0)
- dw = np.dot(X.T, upstream grad)
- dx = np.dot(upstream grad, w.T)
"""


def affine_forward(x, w, b):
    dimen = x.shape                                      # (N, d_1, ..., d_k)
    X = np.reshape(x, (dimen[0], np.prod(dimen[1:])))    # reshape x into shape [N, D]
    out = X.dot(w) + b                                   # out - the result of forward pass
    cache = (x, w, b)                                    # cache - the input value of the layer
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    db = np.sum(dout, axis=0)

    dimen = x.shape                                      # (N, d_1, ..., d_k)
    X = np.reshape(x, (dimen[0], np.prod(dimen[1:])))   # reshape x into shape (N, D)
    dw = np.dot(X.T, dout)                               # (D, M)
    dx = np.reshape(np.dot(dout, w.T), dimen)            # (N, d_1, ..., d_k)
    return dx, dw, db


"""
Detail of a ReLU Layer.

Layer inputs:
- x: any shape

Layer output:
- out: the same shape with input x

forward: 
- x[x < 0] = 0

backward:
- upstream_grad[x < 0] = 0
"""


def relu_forward(x):
    out = x
    out[out < 0] = 0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = dout, cache
    dx[x < 0] = 0
    return dx


"""
Detail of batch normalization:

mean & var:
- sample mean / var stands for the mean and variance of the current mini batch in train mode.
- running mean / var is what we use in test mode.
- running mean / var is the exponentially decaying average of current and previous mean / var.
    - running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    - running_var = momentum * running_var + (1 - momentum) * sample_var

Layer input:
- x: Data of shape (N, D)
- gamma: Scale parameter of shape (D,)
- beta: Shift parameter of shape (D,)
- bn_param: Dictionary with the following keys:
  - mode: 'train' or 'test'; required
  - eps: a constant to avoid the sample variance being zero.
  - momentum: Constant for running mean / variance.
  - running_mean: Array of shape (D,) giving running mean of features
  - running_var Array of shape (D,) giving running variance of features

Layer output:
- out: same shape with x. shape (N, D)

Cache:
- sample_mean, sample_var, x_minus_mean
- sqrt_var = np.sqrt(sample_var + eps)
- div, multi, gamma, beta, eps, x

Forward pass:
- train mode:
    - sample_mean = np.mean(x, axis=0)    # shape (D, )
    - sample_var  = np.var(x, axis=0)     # shape (D, )
    - out = (x - sample_mean)/np.sqrt(sample_var+eps) * gamma + beta
    - running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    - running_var  = momentum * running_var  + (1 - momentum) * sample_var
- test mode
    - out = (x-running_mean)/np.sqrt(running_var+eps) * gamma + beta

Backward pass:
- see more detail in notes
"""


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))  # if 'running_mean' already exist, return its value
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)    # shape (D, )
        sample_var  = np.var(x, axis=0)     # shape (D, )
        sqrt_var = np.sqrt(sample_var+eps)  # shape (D, )

        x_minus_mean = x - sample_mean      # size of (N,D)
        div = x_minus_mean/sqrt_var         # normalize the data

        multi = div * gamma
        out = multi + beta                  # scale and shift
        
        # update the running mean and var
        if not(np.any(running_mean)):       # meaning the first train
            running_mean = sample_mean
            running_var = sample_var
        else:
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var  = momentum * running_var  + (1 - momentum) * sample_var

        cache = (sample_mean,  sample_var, 
                 x_minus_mean, sqrt_var, 
                 div, multi, gamma, beta, eps, x)           # middle values for backpropagation
    elif mode == 'test':
        out = (x-running_mean)/np.sqrt(running_var+eps)     # normalize the data
        out = out * gamma + beta                            # scale and shift parameter
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    This is an implementation in the order of propagation of the computational graph.
    The backward propagation direction follows from dout → ddiv → d(x-μ) → dx
    """
    (N,D) = dout.shape
    ones_matrx = np.ones_like(dout)                                         # to transform shape(D, ) into shape(N, D)

    # intermediates
    (sample_mean,  sample_var,
     x_minus_mean, sqrt_var, 
     div, multi, gamma, beta, eps, x) = cache

    # backpropagation with β and γ from dout to div
    dbeta = np.sum(dout, axis=0)                                            # shape(D, )
    dgamma = np.sum(div*dout, axis=0)                                       # shape(D, )
    ddiv = dout*gamma                                                       # shape(N, D)

    # backpropagation from div to (x-μ)
    # (x-μ) has two branches: (x-μ)_1 and  (x-μ)_2
    dx_minus_mean_1 = 1 / sqrt_var * ddiv                                   # shape(N, D)
    dsqrt_var = np.sum(x_minus_mean * ddiv, axis=0) * (-1 / (sqrt_var**2))  # shape(D, )
    dvar = dsqrt_var / (2 * sqrt_var)                                   # shape(D, )
    dx_minus_mean_2 = ones_matrx * dvar * 2 / N * x_minus_mean              # shape(N, D)
    dx_minus_mean = dx_minus_mean_1 + dx_minus_mean_2                       # shape(N, D)

    # backpropagation form (x-μ) to x
    dmean = np.sum(-dx_minus_mean, axis=0)                                  # shape(D, )
    dx_1 = ones_matrx * dmean / N                                           # shape(N, D)
    dx_2 = dx_minus_mean                                                    # shape(N, D)
    dx = dx_1 + dx_2                                                        # shape(N, D)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    This is an simplified version of the batchnorm_backward.
    It simplifies the equations in batchnorm_backward and combines them.
    The input and output are the same with batchnorm_backward.
    """
    (N,D) = dout.shape

    # intermediates
    (sample_mean,  sample_var,
     x_minus_mean, sqrt_var,
     div, multi, gamma, beta, eps, x) = cache

    # backpropagation with β and γ from dout to div
    dbeta = np.sum(dout, axis=0)                                            # shape(D, )
    dgamma = np.sum(div*dout, axis=0)                                       # shape(D, )

    dx_minus_mean = gamma/sqrt_var *(dout + x_minus_mean/N * np.sum( - x_minus_mean/(sqrt_var**2) * dout, axis=0))
    dx = dx_minus_mean - 1/N * np.sum(dx_minus_mean, axis=0)
    return dx, dgamma, dbeta


"""
Detail of dropout layer:

Layer input:
- x: Data of any shape
- p: (1-p) represents the proportion of zeros

Layer output:
- out: same shape with x with (1-p) set to zero

Forward pass:
- train mode:
    - mask = np.random.rand(* x.shape) < p
    - out = x * mask

- test mode
    - out = x * p

Backward pass:
- dx = dout * mask
"""


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    # 1-p is the proportion of the '0' in mask

    mask = None
    out = None

    if mode == 'train':
        mask = np.random.rand(* x.shape) < p
        out = x * mask
    elif mode == 'test':
        out = x * p
    cache = (dropout_param, mask)

    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    p, mode = dropout_param['p'], dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout * p
    return dx


"""
A naive implementation of the forward pass for a convolutional layer.

Layer Input:
- x: Input data of shape (N, C, H, W)
- w: Filter weights of shape (K, C, F_H, F_W)
- b: Biases, of shape (K,)
- conv_param: A dictionary with the following keys:
  - 'stride': default 1
  - 'pad': default 0

Layer Output:
- out: Output data, of shape (N, F, H', W') where H' and W' are given by
  H' = 1 + (H + 2 * pad - HH) // stride
  W' = 1 + (W + 2 * pad - WW) // stride
  
Cache:
- cache: (x, w, b, conv_param)
"""


def conv_forward_naive(x, w, b, conv_param):
    (num_imgs, img_channels, img_height, img_width) = x.shape   # img size: x ∈ N × C × H × W
    (ksize, F_channels, F_height, F_width) = w.shape            # filter size
    if img_channels != F_channels:
        raise ValueError ('img channel should equal to the kernel channel!')

    pad = conv_param['pad']  # the padding parameter
    stride = conv_param['stride']  # the stride parameter

    # out_size = int((in_size - filter_size + 2 * pad) / stride + 1) where int is floor
    newImgHeight = int(((img_height + 2 * pad - F_height) / stride) + 1)
    newImgWidth = int(((img_width + 2 * pad - F_width) / stride) + 1)

    # zero padding
    x_pad = np.zeros((num_imgs, img_channels, img_height + 2 * pad, img_width + 2 * pad))
    # (0+pad):(size+2*pad-pad) is where the original data located
    x_pad[:, :, pad : img_height + pad, pad : img_width + pad] = x

    # img2col: a method to accelerate the convolution
    # img2col squeezes each Receptive Field into a vector and store in matrix 'cols'
    # Using img2col, the convolution result is 'weights.dot(cols)'
    cols = np.zeros((F_height * F_width * F_channels, newImgHeight * newImgWidth * num_imgs))
    for num in range(num_imgs):             # for each image
        for hei in range(newImgHeight):     # At a given height, iterate through all the widths
            for wid in range(newImgWidth):  # At a given height and a given weight, what is used to do convolution?
                # voxel represents the field used to do convolution
                # vector is the vector form of voxel
                current_voxel = x_pad[num, :, hei*stride : F_height+hei*stride, wid*stride : F_width+wid*stride]
                current_vector = current_voxel.reshape(F_height * F_width * F_channels, )
                cols[:, num * newImgWidth * newImgHeight + hei * newImgWidth + wid] = current_vector

    conv_param['cols'] = cols  # will be used in back propagation

    weights = w.reshape(ksize, F_height * F_width * img_channels)
    out_col = weights.dot(cols)
    out_col_b = out_col + b[:, np.newaxis]

    # col2img:
    # out_col_b: k × (newImgHeight * newImgWidth * num_imgs)
    # out_img: num_imgs × ksize × newImgHeight × newImgWidth
    # each column of out_col_b (size k) is img[x, :, y, z]
    # where x y z are determined by which column it is
    out_img = np.zeros((num_imgs, ksize, newImgHeight, newImgWidth))
    for loc in range(newImgHeight * newImgWidth * num_imgs):
        which_img = loc // (newImgHeight * newImgWidth)                 # which image
        inter_loc = loc % (newImgHeight * newImgWidth)                  # which pixel

        heig_loc = inter_loc // newImgWidth                             # height location
        wid_loc = inter_loc % newImgWidth                               # width location
        out_img[which_img, :, heig_loc, wid_loc] = out_col_b[:, loc]    # assign

    out = out_img
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    (num_imgs, img_channels, img_height, img_width) = x.shape  # input data information
    (ksize, img_channels, F_height, F_width) = w.shape  # size information about the filter
    cols = conv_param['cols']  # added one of the previous calculated variables, so that no need to calculate it again.
    pad = conv_param['pad']  # the padding parameter
    stride = conv_param['stride']  # the stride parameter

    (num_imgs, ksize, newImgHeight, newImgWidth) = dout.shape

    # dout → dout_col_b
    dout_col_b = np.zeros((ksize, newImgHeight * newImgWidth * num_imgs))
    for num in range(num_imgs):
        img = dout[num, :, :, :]
        vector = img.reshape(ksize, newImgHeight * newImgWidth)
        dout_col_b[:, num * newImgHeight * newImgWidth:(num + 1) * newImgHeight * newImgWidth] = vector

    # dout_col_b → dcols
    db = np.sum(dout_col_b, axis=1)  # out_col_b = out_col + b

    r_weight = w.reshape(ksize, F_height * F_width * img_channels)  # reshaped weight
    dr_weights = dout_col_b.dot(cols.T)  # derivative of:  out_col  = r_weights * cols
    dw = dr_weights.reshape(ksize, img_channels, F_height, F_width)
    dcols = (r_weight.T).dot(dout_col_b)  # derivative of:  out_col  = r_weights * cols

    # dcols → dx_pad
    dx_pad = np.zeros((num_imgs, img_channels, img_height + 2 * pad,
                       img_width + 2 * pad))  # after zero padding, what is the size should be for the padded images

    for loc in range(dcols.shape[1]):  # iterate through each column in the dcols
        which_img = loc // (newImgHeight * newImgWidth)  # which new image this loc is belong to
        whcih_pix = loc % (newImgHeight * newImgWidth)  # which pixel is within the current new image

        hei_loc = whcih_pix // newImgWidth  # the height index in the new image
        wid_loc = whcih_pix % newImgWidth  # the width index in the new image

        temp = dcols[:, loc].reshape(img_channels, F_height, F_width)
        dx_pad[which_img, :, hei_loc * stride:F_height + hei_loc * stride,
        wid_loc * stride:F_width + wid_loc * stride] += temp

    # dx_pad → dx
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


"""
Forward pass for layer normalization.

During both training and test-time, the incoming data is normalized per data-point.
Unlike the batch normalization, layer normalization doesn't need to keep track of 
averages or something else. Therefore, there is no difference between train and test.

Layer Input:
- x: Data of shape (N, D)
- gamma: Scale parameter of shape (D,)
- beta: Shift paremeter of shape (D,)
- ln_param: Dictionary with the following keys:
    - eps: Constant for numeric stability

Layer Output:
- out: of shape (N, D)

Cache:
- cache: (sample_mean,  sample_var, 
             x_minus_mean, sqrt_var, 
             y, multi, gamma, beta, eps, x)
"""


def layernorm_forward(x, gamma, beta, ln_param):
    eps = ln_param.get('eps', 1e-5)

    sample_mean = np.mean(x, axis=1)              # shape: (N, )
    sample_var  = np.var(x, axis=1)               # shape: (N, )

    x_minus_mean = x - sample_mean[:,np.newaxis]  # shape: (N,D)
    
    sqrt_var = np.sqrt(sample_var+eps)            # shape: (N, )
    
    y = x_minus_mean/sqrt_var[:,np.newaxis]       # layer normalization
    multi = y * gamma           
    out = y * gamma + beta                        # scale and shift

    cache = (sample_mean,  sample_var, 
             x_minus_mean, sqrt_var, 
             y, multi, gamma, beta, eps, x)       # preparation for back propagation

    return out, cache


def layernorm_backward(dout, cache):
    (N, D) = dout.shape
    ones_matrx = np.ones_like(dout)  # later used to expand (N, ) matrix into (N, D) matrix
         
    (sample_mean,  sample_var, 
     x_minus_mean, sqrt_var, 
     y, multi, gamma, beta, eps, x) = cache 

    # backpropagation with (γ) and (β) from (dout) to (dy)
    dbeta = np.sum(dout, axis=0)                                            # shape: (D,)
    dgamma = np.sum(y * dout, axis=0)                                       # shape: (D,)
    dy = gamma * dout                                                       # shape: (N, D)
    
    # backpropagation from (y) to (x-μ)
    dx_minus_mean_1 = dy / sqrt_var[:, np.newaxis]                          # shape: (N, D)
    dsqrt_var = np.sum(x_minus_mean * dy, axis=1) * (-1 / (sqrt_var**2))    # shape: (N, )
    dvar = dsqrt_var / (2 * sqrt_var)                                       # shape: (N, )
    dx_minus_mean_2 = ones_matrx * dvar / D * 2 * x_minus_mean              # shape: (N, D)
    dx_minus_mean = dx_minus_mean_1 + dx_minus_mean_2                       # shape: (N, D)

    # backpropagation from (x-μ) to x
    dx_1 = dx_minus_mean                                                    # shape: (N, D)
    dmean = np.sum(-dx_minus_mean, axis=1)                                  # shape: (N, )
    dx_2 = ones_matrx * dmean / D                                           # shape: (N, D)
    dx = dx_1 + dx_2                                                        # shape: (N, D)

    return dx, dgamma, dbeta


"""
A naive implementation of the forward pass for a max-pooling layer.

Layer Inputs:
- x: Input data, of shape (N, C, H, W)
- pool_param: dictionary with the following keys:
  - 'pool_height': The height of each pooling region
  - 'pool_width': The width of each pooling region
  - 'stride': The distance between adjacent pooling regions

Layer Outputs:
- out: of shape (N, C, H', W') where H' and W' are given by
  H' = 1 + (H - pool_height) / stride
  W' = 1 + (W - pool_width) / stride
  
Cache:
- cache: (x, pool_param, out_index)
"""


def max_pool_forward_naive(x, pool_param):
    (N, C, H, W) = x.shape               # input shape: (N, C, H, W)
    pool_w = pool_param['pool_width']    # max pooling parameters
    pool_h = pool_param['pool_height']
    stride = pool_param['stride']

    new_h = int((H-pool_h)/stride + 1)  # height of output
    new_w = int((W-pool_w)/stride + 1)  # width of output
    
    out = np.zeros((N, C, new_h, new_w))        # output shape
    out_index = np.zeros((N, C, new_h, new_w))  # track where the max value is located in its corresponding voxel
    for h in range(new_h):
        for w in range(new_w):                                  # find the voxel for out[:, :, h, w]
            voxel = x[:, :, h*stride: pool_h+h*stride, w*stride:pool_w+w*stride]
            r_voxel = voxel.reshape(N, C, pool_w*pool_h)        # reshape to find the max and the index
            out[:, :, h, w] = np.max(r_voxel, axis=2)
            out_index[:, :, h, w] = np.argmax(r_voxel, axis=2)
    cache = (x, pool_param, out_index)
    return out, cache


def max_pool_backward_naive(dout, cache):
    (x, pool_param, out_index) = cache
    (N, C, new_h, new_w) = dout.shape    # upstream grad shape
    pool_w = pool_param['pool_width']    # max pooling parameters
    pool_h = pool_param['pool_height']
    stride = pool_param['stride']
    
    dx = np.zeros_like(x)                # our desired grad

    for h in range(new_h):
        for w in range(new_w):              # go through all positions of the upstream grad
            maxID = out_index[:, :, h, w]   # location of the max value at position (h, w), shape (N, C)
            inter_h = np.array(maxID // pool_w, dtype=int)
            inter_w = np.array(maxID % pool_w, dtype=int)

            dvoxel = np.zeros((N, C, pool_h, pool_w))     # the grad corresponding to position (h, w)
            for n in range(N):
                for c in range(C):                        # go through N samples and C channels
                    dvoxel[n, c, inter_h[n, c], inter_w[n, c]] = dout[n, c, h, w]       # update dvoxel
            dx[:, :, h*stride: pool_h+h*stride, w*stride:pool_w+w*stride] += dvoxel     # update dx
    return dx


"""
spatial batch normalization
While batch normalization normalizes (N, D) data where D is the length of data,
spatial batch normalization normalizes (···, C) where C is the channel number. 
By simply transposing the input data and calling batchnorm function, 
spatial_batchnorm could be achieved.

Layer Inputs:
- x: Input data of shape (N, C, H, W)
- gamma: Scale parameter, of shape (C,)
- beta: Shift parameter, of shape (C,)
- bn_param: Dictionary with the following keys:
  - mode: 'train' or 'test'; required
  - eps: Constant for numeric stability
  - momentum: Constant for running mean / variance. momentum=0 means that
    old information is discarded completely at every time step, while
    momentum=1 means that new information is never incorporated. The
    default of momentum=0.9 should work well in most situations.
  - running_mean: Array of shape (C,) giving running mean of features
  - running_var Array of shape (C,) giving running variance of features

Layer Outputs:
- out: Output data, of shape (N, C, H, W)
"""


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)  # transpose changes x into (N, H, W, C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)    # back to (N, C, H, W)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


"""
Computes the forward pass for spatial group normalization.
In contrast to layer normalization, group normalization splits each entry 
in the data into G contiguous pieces, which it then normalizes independently.
Per feature shifting and scaling are then applied to the data, in a manner identical 
to that of batch normalization and layer normalization.

Inputs:
- x: Input data of shape (N, C, H, W)
- gamma: Scale parameter, of shape (C,)
- beta: Shift parameter, of shape (C,)
- G: Integer mumber of groups to split into, should be a divisor of C
- gn_param: Dictionary with the following keys:
  - eps: Constant for numeric stability

Returns a tuple of:
- out: Output data, of shape (N, C, H, W)
- cache: Values needed for the backward pass
"""


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    eps = gn_param.get('eps', 1e-5)
    N, C, H, W = x.shape
    x_group = x.reshape(N, G, C//G, H, W)                     # divide x into G groups along channel direction
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)    # mean: (N, G, 1, 1, 1)
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)      # var: (N, G, 1, 1, 1)
    x_groupnorm = (x_group-mean)/np.sqrt(var+eps)             # normalization
    x_norm = x_groupnorm.reshape(N, C, H, W)                  # back to (N, C, H, W)
    out = x_norm * gamma + beta                               # scale & shift
    cache = (G, x, x_norm, mean, var, beta, gamma, eps)
    return out, cache



def spatial_groupnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    G, x, x_norm, mean, var, beta, gamma, eps = cache
    x_group = x.reshape((N, G, C // G, H, W))

    # from dout to dbeta & dgamma
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)

    # from dout to dx_groupnorm
    dx_norm = dout * gamma
    dx_groupnorm = dx_norm.reshape((N, G, C // G, H, W))

    # from dx_groupnorm to dx_group_minus_mean: 2 branches
    dx_group_minus_mean_1 = dx_groupnorm / np.sqrt(var + eps)
    dvar = np.sum(dx_groupnorm * (x_group - mean) * (-1/2) * (var + eps)**(-3/2), axis=(2, 3, 4), keepdims=True)
    N_GROUP = C // G * H * W
    dx_group_minus_mean_2 = dvar * np.ones_like(x_group) / N_GROUP * 2 * (x_group - mean)
    dx_group_minus_mean = dx_group_minus_mean_1 + dx_group_minus_mean_2

    # from dx_group_minus_mean to dx_group: 2 branches
    dx_group_1 = dx_group_minus_mean
    dmean = np.sum(-dx_group_minus_mean, axis=(2, 3, 4), keepdims=True)
    dx_group_2 = dmean * np.ones_like(x_group) / N_GROUP
    dx_group = dx_group_1 + dx_group_2

    # from dx_group to dx
    dx = dx_group.reshape(N, C, H, W)
    return dx, dgamma, dbeta




def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)   # record the number of nonzero loss in each row of margins
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    # each nonzero loss contributes a (-1) to the grad at label position
    dx[np.arange(N), y] = dx[np.arange(N), y] - num_pos
    dx = dx / N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_x = x - np.max(x, axis=1, keepdims=True)        # To avoid overflow
    Z = np.sum(np.exp(shifted_x), axis=1, keepdims=True)    # Σe^(···)
    log_probs = shifted_x - np.log(Z)                       # log(e^(···)/Σe^(···))
    probs = np.exp(log_probs)                               # e^(···)/Σe^(···)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
