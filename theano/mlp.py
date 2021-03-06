"""Multi-layer perceptron training script."""

import theano
import theano.tensor as T

import lasagne
from lasagne.init import Constant, Normal

from lib.datasets.creditcard import load_dataset
from lib.datasets.base_dataset import iterate_minibatches


def build_mlp(input_var, layer_widths, input_dropout=.2, hidden_dropout=.5):
    """Build a multi-layer perceptron model with the given layer widths."""

    # Input layer and dropout:
    network = lasagne.layers.InputLayer(shape=(500, 30), input_var=input_var)
    if input_dropout:
        network = lasagne.layers.dropout(network, p=input_dropout)

    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for width in layer_widths:
        network = lasagne.layers.DenseLayer(
            network, width, nonlinearity=nonlin, W=Normal(), b=Constant(val=.1))
        if hidden_dropout:
            network = lasagne.layers.dropout(network, p=hidden_dropout)

    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 2, nonlinearity=softmax)
    return network


def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    network = build_mlp(input_var, [30, 45, 67, 100])

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

    # Create a loss expression for validation/testing. The crucial difference here is that
    # we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(
        T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    for epoch in range(num_epochs):
        print('Epoch %d' % (epoch + 1))
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        # start_time = time.time
        for batch in iterate_minibatches(X_train, Y_train, 500, shuffle=True, progress=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        # print("Epoch {} of {} took {:.3f}s".format(
        #     epoch + 1, num_epochs, time.time() - start_time))
        print(f"  training loss:\t\t{train_err / train_batches:.6f}")
        print(f"  validation loss:\t\t{val_err / val_batches:.6f}")
        print(f"  validation accuracy:\t\t{val_acc / val_batches * 100:.2f} %")

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print(f"  test loss:\t\t\t{test_err / test_batches:.6f}")
    print(f"  test accuracy:\t\t{test_acc / test_batches * 100:.2f} %")

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main(50)
