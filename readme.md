This is the official implementation of GBDR

Due to the ongoing nature of our research, please do not disclose our code! Thank you for your cooperation.

Below, we take the defense against the BadNet attack on the CIFAR10 dataset as an example to demonstrate the execution process of our code.

1. Modify the variable in the "global_param.py" file to the absolute path of the "codes" folder on your computer, for example, "/your_path/codes".

2. Generate the poisoned backdoor dataset and train the backdoor target model. Enter the "attack" folder.

   ```python
   python badnet.py -y config/attack/badnet/bd_cifar10_WRN28.yaml
   ```
The attack  results will be saved in "record/attack/badnet/bd_cifar10_WRN28".

3. Defense the attack. Enter the "defense" folder.

   ```python
   python gbdr.py -ya config/attack/badnet/bd_cifar10_WRN28.yaml -yd config/defense/gbdr/cifar10.yaml -p record/defense/gbdr/badnet_cifar10_WRN28 -dv cuda:0
   ```
The defense  results will be saved in "record/defense/gbdr/badnet_cifar10_WRN28".
