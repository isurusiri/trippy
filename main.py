# Deep Dream experiment
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile


def main():
    # This code uses a pre-trained convolutional neural network,
    # which is trained by google brain team (google).
    # The code below is to download google's pre-trained neural network and
    # extract it if it's already not available in trained_data directory.
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = './trained_data/'
    model_name = os.path.split(url)[-1]
    training_data = os.path.join(data_dir, model_name)
    if not os.path.exists(training_data):
        print('Training data is not available in local directory \n start downloading training data...')
        model_path = urllib.request.urlopen(url)
        with open(training_data, 'wb') as output:
            output.write(model_path.read())
            print('Successfully downloaded training data')
        with zipfile.ZipFile(training_data, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            print('Unzipped training data')

    # creating a gray image with noise
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

    # load the tensorflow graph containing the convolutional neural network model
    # trained by google.
    model_fn = 'tensorflow_inception_graph.pb'

    # Creates a new tensorflow graph and a session.
    # Load the pre-trained model to new tensorflow graph.
    # Read the loaded graph by parsing it.
    # crates an input tensor with a placeholder.
    # defines the mean value of images pixels, to help with feature learning.
    # import the graph definitions with preprocessed tensor.
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    input_tensor = tf.placeholder(np.float32, name='input')
    mean_pixels = 117.0
    preprocessed_tensor = tf.expand_dims(input_tensor - mean_pixels, 0)
    tf.import_graph_def(graph_def, {'input': preprocessed_tensor})
    print('Graph loaded successfully')

    # Loads the layers in the model to an array.
    # Store all the features passed across layers in an array.
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
    print('Number of layers', len(layers))
    print('Total number of feature channels', sum(feature_nums))

    # Strip large constant values from graph_def.
    def stripConsts(graph_def, max_const_size=32):
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>" % size
        return strip_def

    def renameNodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
        return res_def

    def showArray(a):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.show()

    # Normalize the image range for visualization
    def visualizer(a, s=0.1):
        return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

    # Helper for getting layer output tensor
    def T(layer):
        return graph.get_tensor_by_name("import/%s:0" % layer)

    # Defines the optimization objective
    # Normalizes the gradient, so the same step size should work
    def renderNaive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, input_tensor)[0]

        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {input_tensor: img})
            img += g * step
        showArray(visualizer(img))

    # Transforms  tensorflow graph generating function into a regular one.
    def tensorflowGraphTransformer(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tensorflowGraphTransformer(np.float32, np.int32)(resize)

    # Compute the value of tensor t_grad over the image in a tiled way.
    # Random shifts are applied to the image to blur tile boundaries over
    # multiple iterations.
    def calcGradTiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {input_tensor: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def renderDeepDream(t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, input_tensor)[0]

        # splits the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for _ in range(iter_n):
                g = calcGradTiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))

        showArray(img / 255.0)



    # Pick a layer to enhance image and a feature channel to visualize
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 155

    # open image to be tripped with ;)
    img0 = PIL.Image.open('landscape.jpg')
    img0 = np.float32(img0)

    # Apply gradient ascent to decided layer
    renderDeepDream(tf.square(T('mixed4c')), img0)


if __name__ == '__main__':
    main()