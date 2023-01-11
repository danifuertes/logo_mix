import os
import time
import pickle
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from utils.utils import get_lines, get_anchors, get_classes, get_model, HistoryCallback
from utils.data_utils import data_generator
from options import get_options
# os.environ['KMP_WARNINGS'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(opts):

    # Create save_dir if not exists and restore_model == False
    time_txt = time.strftime("%Y%m%dT%H%M%S")
    if not opts.restore_model:
        opts.save_dir = os.path.join(opts.save_dir, 'YoloV3_{}_{}'.format(opts.dataset_name.replace('/', '_'),
                                                                          time_txt))
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)

    # Print and save options
    print('\nOptions:')
    log_dir = os.path.join(opts.save_dir, 'log_dir')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_opts = open(os.path.join(log_dir, 'options_{}.txt'.format(time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()

    # Load file containing the path of the training and validation images
    lines_train = get_lines(opts.train_path, opts.dataset_path)
    num_train = len(lines_train)
    if os.path.isfile(opts.val_path):
        lines_val = get_lines(opts.val_path, opts.dataset_path)
        num_val = len(lines_val)
    else:
        num_val = int(num_train * opts.val_perc)
        num_train -= num_val
        lines_val = lines_train[num_train:]
        lines_train = lines_train[:num_train]

    # Input image shape
    input_shape = (opts.image_width, opts.image_height)

    # Get classes and anchors
    class_names = get_classes(opts.classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(opts.anchors_path, num_anchors=opts.num_clusters, new_anchors=opts.new_anchors,
                          use_bb=opts.use_bb)

    # Create model
    tiny_yolo = len(anchors) == 6 if opts.use_bb else False  # Tiny version of YoloV3 (faster but no so accurate)
    if not opts.weights_path == 'yolo3/yolo_weights.h5':
        weights_path = os.path.join(opts.save_dir, opts.weights_path) if opts.restore_model else ''
    else:
        weights_path = opts.weights_path
    th = opts.i_th if opts.use_bb else opts.d_th / max(*input_shape)
    model = get_model(input_shape, num_classes, anchors=anchors, use_bb=opts.use_bb, restore_model=opts.restore_model,
                      weights_path=weights_path, freeze_body=2, tiny_yolo=tiny_yolo, num_gpu=opts.num_gpu, iou_th=th)

    # Callbacks
    hist_callback = HistoryCallback(log_dir)
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(os.path.join(opts.save_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=opts.reduce_lr_period, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=opts.early_stop_period, verbose=1)
    callbacks = [hist_callback, logging, checkpoint, reduce_lr]

    # Define yolo loss
    loss = 'yolo_loss' if opts.use_bb else 'yolo_loss_single_point'

    # Train stages (freeze / unfreeze layers)
    train_stages = {}
    if opts.train_frozen:
        train_stages['frozen'] = {
            'batch': opts.batch_frozen,
            'lr': opts.lr_frozen,
            'epochs': opts.epochs_frozen,
            'initial_epoch': opts.initial_epoch
        }
    if opts.train_unfrozen:
        train_stages['unfrozen'] = {
            'batch': opts.batch_unfrozen,
            'lr': opts.lr_unfrozen,
            'epochs': opts.epochs_unfrozen,
            'initial_epoch': opts.initial_epoch if opts.restore_model or not opts.train_frozen \
                                                else opts.epochs_frozen + opts.initial_epoch
        }
    assert len(train_stages) > 0, "Either train_frozen or train_unfrozen must be True"

    # Train with frozen layers first to get a stable loss. Then, unfreeze and continue training
    for k, stage in train_stages.items():

        # Unfreeze layers and maybe add early stop callback
        if k == 'unfrozen':
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            if opts.early_stop:
                callbacks.append(early_stopping)

        # Compile model
        model.compile(optimizer=Adam(lr=stage['lr']), loss={loss: lambda y_true, y_pred: y_pred})
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, stage['batch']))

        # Data generators
        train_gen = data_generator(lines_train, stage['batch'], input_shape, num_classes, anchors, use_bb=opts.use_bb,
                                   data_aug=opts.data_aug, use_logomix=opts.use_logomix, logomix_perc=opts.logomix_perc,
                                   use_attentive=opts.use_attentive, fake_box=(opts.fake_width, opts.fake_height))
        val_gen = data_generator(lines_val, stage['batch'], input_shape, num_classes, anchors, use_bb=opts.use_bb)

        # Fit
        try:
            model.fit_generator(train_gen,
                                steps_per_epoch=max(1, num_train // stage['batch']),
                                validation_data=val_gen,
                                validation_steps=max(1, num_val // stage['batch']),
                                epochs=stage['epochs'],
                                initial_epoch=stage['initial_epoch'],
                                callbacks=callbacks,
                                use_multiprocessing=True,
                                max_queue_size=10,
                                workers=16)
        except KeyboardInterrupt:
            model.save_weights(os.path.join(opts.save_dir, 'weights_{}_XXX.h5'.format(k)))
            return
        model.save_weights(os.path.join(opts.save_dir, 'final_{}.h5'.format(k)))

        # Load history
        if os.path.isfile(os.path.join(log_dir, 'history.pkl')):
            with open(os.path.join(log_dir, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)

            # Plot history
            elements = ['loss']
            for element in elements:
                hist_train, hist_val = [], []
                for k, v in history.items():
                    hist_train.append(v[element])
                    hist_val.append(v['val_' + element])
                plt.plot(hist_train)
                plt.plot(hist_val)
                plt.title('model ' + element)
                plt.ylabel(element)
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(os.path.join(opts.save_dir, 'log_dir', element + '.png'))
                plt.show()
                plt.clf()
        print('Finished')


if __name__ == '__main__':
    main(get_options())
