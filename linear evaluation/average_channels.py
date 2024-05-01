import os
import numpy as np

"""Average R and r2 across all channels and save it as a new channel AVR"""

def create_avr_channels(checkpoint_dir):
    for subject in os.listdir(checkpoint_dir):
        subject_dir = os.path.join(checkpoint_dir, subject)
        if os.path.isdir(subject_dir):
            for layer in os.listdir(subject_dir):
                print(layer)
                layer_dir = os.path.join(subject_dir, layer)
                if os.path.isdir(layer_dir):
                    avr_dir = os.path.join(layer_dir, "AVR")
                    os.makedirs(avr_dir, exist_ok=True)  # Create AVR directory if it doesn't exist
                    test_rs_sum = None
                    test_r2s_sum = None
                    num_channels = 0
                    for channel in os.listdir(layer_dir):
                        channel_dir = os.path.join(layer_dir, channel)
                        if os.path.isdir(channel_dir):
                            test_rs_file = os.path.join(channel_dir, "test_rs.npy")
                            test_r2s_file = os.path.join(channel_dir, "test_r2s.npy")
                            if os.path.isfile(test_rs_file) and os.path.isfile(test_r2s_file):
                                test_rs = np.load(test_rs_file)
                                test_r2s = np.load(test_r2s_file)
                                if test_rs_sum is None:
                                    test_rs_sum = test_rs
                                    test_r2s_sum = test_r2s
                                else:
                                    test_rs_sum += test_rs
                                    test_r2s_sum += test_r2s
                                num_channels += 1
                    if num_channels > 0:
                        avr_test_rs = test_rs_sum / num_channels
                        avr_test_r2s = test_r2s_sum / num_channels
                        np.save(os.path.join(avr_dir, "test_rs.npy"), avr_test_rs)
                        np.save(os.path.join(avr_dir, "test_r2s.npy"), avr_test_r2s)


output_dir_base = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\EEG data\linreg final outputs"
run_dirs = ["OADS", "OADSx30", "OADSx6", "STL10", "Supervised ResNet18 + OADS OC", "Untrained"]

for run_dir in run_dirs:
    checkpoint_dir = os.path.join(output_dir_base, run_dir)
    create_avr_channels(checkpoint_dir)
    print(run_dir)
