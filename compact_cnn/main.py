# Kapre version >0.0.2.3 (float32->floatx fixed version)
from argparse import Namespace
import models as my_models
from keras import backend as K
import pdb
import numpy as np
from prepare_audio import compute_samples
from glob import glob
import re, os
import argparse
K.set_image_dim_ordering('th')
import pandas as pd
from datetime import datetime

TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic',
            'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
            'beautiful', 'metal', 'chillout', 'male vocalists',
            'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
            '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
            'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
            'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
            '70s', 'party', 'country', 'easy listening',
            'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
            'Progressive rock', '60s', 'rnb', 'indie pop',
            'sad', 'House', 'happy']

def main(mode, conv_until=None):
    # setup stuff to build model

    # This is it. use melgram, up to 6000 (SR is assumed to be 12000, see model.py),
    # do decibel scaling
    assert mode in ('feature', 'tagger')
    if mode == 'feature':
        last_layer = False
    else:
        last_layer = True

    if conv_until is None:
        conv_until = 4


    print(f'K.image_dim_ordering(): {K.image_dim_ordering()}')
    assert K.image_dim_ordering() == 'th', ('image_dim_ordering should be "th". ' +
                                            'open ~/.keras/keras.json to change it.')

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.

    model = my_models.build_convnet_model(args=args, last_layer=last_layer)
    model.load_weights('weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                       by_name=True)
    model.layers[1].summary()
    model.summary()
    # and use it!
    return model


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Music Auto tagger')
    parser.add_argument('--folder', type=str,
                        help='path to the audio_files')
    parser.add_argument('--mode', default='tagger', type=str,
                        help='Specify if you want to tag or compute feature')
    parser.add_argument('--output-dir', default='data/feature_dataframes', type=str,
                        help='Output dir where to store the features dataframe. only used if mode="feature"')
    parser.add_argument('--with-sigmoid', default=False, type=bool,
                        help='apply sigmoid to ')
    
    cmd_args = parser.parse_args()
    audio_paths = glob(os.path.join(cmd_args.folder, '*'))
    
    audio_paths = [x for x in audio_paths if re.search('[.](mp3|wav)$', x) is not None]
    input_array = compute_samples(audio_paths, sr=12000, duration=29, mono=True)
    
    # main('tagger') # music tagger
    if cmd_args.mode == 'tagger':
        model = main('tagger')
        predictions = model.predict(input_array)
        print(f'Predictions shape: {predictions.shape}')
        for i in range(min(100, predictions.shape[0])):
            #detected_tags = [TAGS[x] for x in range(predictions.shape[1]) if predictions[i,x]] 
            #print(f'Tags for file {audio_paths[i]}\n: {detected_tags}')
            sorted_result = sort_result(TAGS, predictions[i, :].tolist())
            print(audio_paths[i])
            print(sorted_result[:5])
            print(sorted_result[5:10])
            print(' ')       


    
    else:              
        # load models that predict features from different level
        models = []
        model4 = main('feature')  # equal to main('feature', 4), highest-level feature extraction
        model3 = main('feature', 3)  # low-level feature extraction.
        model2 = main('feature', 2)  # lower-level..
        model1 = main('feature', 1)  # lowerer...
        model0 = main('feature', 0)  # lowererer.. no, lowest level feature extraction.

        # prepare the models
        models.append(model4)
        models.append(model3)
        models.append(model2)
        models.append(model1)
        models.append(model0)


        # get features from each layer and conatenate them
        feat = np.hstack([md.predict(input_array) for md in models])
        feat_df = pd.DataFrame(feat)
        feat_df['track_name'] = [os.path.basename(x) for x in audio_paths]
        output_name = str(datetime.now()).split('.')[0].replace(' ', '_') + '.csv'
        feat_df.to_csv(os.path.join(cmd_args.output_dir, output_name))
        print(f'Files saved to data/feature_dataframes/{output_name}')
        #for i in range(min(5, feat.shape[0])):
        #        print(f'features for file {audio_paths[i]}\n: {predictions[i,:]}')
