This script is to generate the triggers of SSBA.

1. Train the encoder-decoder model

   ```python
   python train.py -y bd_trigger/ssba/config/train_param/cifar10.yaml
   ```

2. Generate the triggers by the trained encoder-decoder model

   ```python
   python embed_fingerprints.py -y bd_trigger/ssba/config/test_param/cifar10.yaml
   ```

3. Then the triggers are in "record/cifar10/train_bd.hdf5" and "record/cifar10/test_bd.hdf5"

